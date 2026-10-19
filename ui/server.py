"""Локальный UI для расшифровки звонков. Запуск: calls-ui (http://127.0.0.1:8790)."""

import os
import queue
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from contextlib import closing
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import call2text as c  # noqa: E402

HOST, PORT = "127.0.0.1", 8790
WINDOWS = sys.platform == "win32"
UI_DIR = Path(__file__).resolve().parent
DATA_DIR = UI_DIR.parent / "data"
DB_PATH = DATA_DIR / "calls.db"
MAX_HOURS = 2

app = FastAPI()
jobs: "queue.Queue[int]" = queue.Queue()
cancelled: set = set()
models = {}


class Cancelled(Exception):
    pass


# --- база ---

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    with closing(db()) as conn, conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                duration REAL,
                recorded_at REAL,
                created_at REAL NOT NULL,
                speakers INTEGER,
                merge INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,          -- queued | running | done | error | cancelled
                stage TEXT,
                progress REAL NOT NULL DEFAULT 0,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS speakers (
                call_id INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                name TEXT NOT NULL,
                idx INTEGER NOT NULL,
                PRIMARY KEY (call_id, label)
            );
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY,
                call_id INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
                start REAL NOT NULL,
                end REAL NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL
            );
        """)
        # старые записи с названием-именем файла — на понятное название
        for row in conn.execute("SELECT id, title, path FROM calls").fetchall():
            if row["title"] == Path(row["path"]).stem:
                title, started = nice_title(row["path"])
                if title != row["title"]:
                    conn.execute("UPDATE calls SET title=?, recorded_at=COALESCE(?, recorded_at) WHERE id=?",
                                 (title, started, row["id"]))
        # после перезапуска незаконченные задачи снова в очередь
        conn.execute("UPDATE calls SET status='queued', stage=NULL, progress=0 "
                     "WHERE status IN ('queued', 'running')")
        for row in conn.execute("SELECT id FROM calls WHERE status='queued' ORDER BY id"):
            jobs.put(row["id"])


MONTHS = "янв фев мар апр мая июн июл авг сен окт ноя дек".split()
TELEMOST = re.compile(r"Telemost (\d{2})\.(\d{2})\.(\d{2}) (\d{2})-(\d{2})-(\d{2})")


def nice_title(path: str):
    """Название для списка и время начала звонка. Для записей Телемоста — «Телемост · 15 окт, 10:30»."""
    stem = Path(path).stem
    m = TELEMOST.search(stem)
    if not m:
        return stem, os.path.getmtime(path) if os.path.exists(path) else None
    d, mo, y, hh, mm, ss = map(int, m.groups())
    started = datetime(2000 + y, mo, d, hh, mm, ss)
    return f"Телемост · {d} {MONTHS[mo - 1]}, {hh:02d}:{mm:02d}", started.timestamp()


def set_status(call_id, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    with closing(db()) as conn, conn:
        conn.execute(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))


# --- фоновая расшифровка ---

def get_models():
    if not models:
        models["pipeline"] = c.load_pipeline()
        models["model"] = c.load_model()
    return models["pipeline"], models["model"]


# доли этапов pyannote в общей полосе: разметка занимает 5–65%, распознавание — 65–100%
STEPS = {"segmentation": (0.05, 0.2), "embeddings": (0.2, 0.65)}


def models_cached() -> bool:
    from huggingface_hub import constants
    hub = Path(constants.HF_HUB_CACHE)
    return all((hub / f"models--{name}").is_dir() for name in
               ("ai-sage--GigaAM-v3", "pyannote--speaker-diarization-community-1", "pyannote--segmentation-3.0"))


def process(call_id):
    with closing(db()) as conn:
        call = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    if call is None or call["status"] != "queued":
        return

    def check():
        if call_id in cancelled:
            raise Cancelled()

    stage = "Загрузка моделей" if models_cached() else "Первый запуск: скачиваю модели (~1,5 ГБ)"
    set_status(call_id, status="running", stage=stage, progress=0, error=None)
    pipeline, model = get_models()
    check()

    set_status(call_id, stage="Чтение записи", progress=0.03)
    wav = c.load_audio(call["path"])
    check()

    def hook(step, artifact, file=None, total=None, completed=None):
        check()
        if total and step in STEPS:
            lo, hi = STEPS[step]
            set_status(call_id, stage="Разметка собеседников", progress=lo + (hi - lo) * completed / total)

    set_status(call_id, stage="Разметка собеседников", progress=0.05)
    turns = c.diarize(wav, call["speakers"], pipeline=pipeline, hook=hook)
    if call["merge"]:
        turns = c.merge_turns(turns)
    turns = [t for t in turns if t[1] - t[0] >= c.MIN_TURN]

    with closing(db()) as conn, conn:
        conn.execute("DELETE FROM segments WHERE call_id=?", (call_id,))
        conn.execute("DELETE FROM speakers WHERE call_id=?", (call_id,))

    names = {}
    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = os.path.join(tmp, "turn.wav")
        for i, (start, end, label) in enumerate(turns):
            check()
            set_status(call_id, stage=f"Распознавание: {i + 1} из {len(turns)}",
                       progress=0.65 + 0.35 * i / len(turns))
            text = c.transcribe_turn(model, wav, start, end, chunk_path)
            if not text:
                continue
            with closing(db()) as conn, conn:
                if label not in names:
                    names[label] = len(names)
                    conn.execute("INSERT INTO speakers VALUES (?, ?, ?, ?)",
                                 (call_id, label, f"Собеседник {len(names)}", names[label]))
                conn.execute("INSERT INTO segments (call_id, start, end, speaker, text) "
                             "VALUES (?, ?, ?, ?, ?)", (call_id, start, end, label, text))

    set_status(call_id, status="done", stage=None, progress=1)


def worker():
    while True:
        call_id = jobs.get()
        stop_awake = keep_awake()  # не даём компьютеру уснуть, пока идёт задача
        try:
            process(call_id)
        except Cancelled:
            set_status(call_id, status="cancelled", stage=None)
        except Exception as exc:  # показываем причину в интерфейсе
            set_status(call_id, status="error", stage=None, error=explain(exc))
        finally:
            stop_awake()
            cancelled.discard(call_id)


def keep_awake():
    if WINDOWS:
        import ctypes
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, действует для этого потока
        return lambda: set_state(0x80000000)
    awake = subprocess.Popen(["caffeinate", "-i"])
    return awake.terminate


def explain(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, KeyError) and "HF_TOKEN" in text:
        if WINDOWS:
            return "Не задан HF_TOKEN. Запустите установку ещё раз, затем перезапустите программу."
        return "Не задан HF_TOKEN в ~/.zprofile. Перезапустите calls-ui из нового окна терминала."
    if isinstance(exc, subprocess.CalledProcessError):
        return "Не удалось прочитать звук из файла. Есть ли в записи звуковая дорожка?"
    if isinstance(exc, MemoryError) or "out of memory" in text.lower():
        return "Не хватило памяти. Закройте тяжёлые приложения и повторите."
    return f"{type(exc).__name__}: {text}"


# --- файлы ---

def probe(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise HTTPException(400, "Это не похоже на аудио- или видеофайл.")


PICK_WINDOWS = """import sys, tkinter, tkinter.filedialog
root = tkinter.Tk(); root.withdraw(); root.attributes("-topmost", True)
print(tkinter.filedialog.askopenfilename(title=sys.argv[1]))"""


def choose_file(prompt: str) -> str:
    if WINDOWS:  # tkinter в отдельном процессе: окну нужен свой главный поток
        out = subprocess.run([sys.executable, "-c", PICK_WINDOWS, prompt], capture_output=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    else:
        script = f'POSIX path of (choose file with prompt "{prompt}")'
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    path = out.stdout.strip()
    if out.returncode != 0 or not path:
        raise HTTPException(400, "Файл не выбран.")
    return os.path.normpath(path)


# --- API ---

class NewCall(BaseModel):
    path: str
    speakers: int | None = None
    merge: bool = True


class Rename(BaseModel):
    name: str


class EditText(BaseModel):
    text: str


class Relink(BaseModel):
    path: str | None = None


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


@app.post("/api/pick")
def pick():
    path = choose_file("Выберите запись звонка")
    return {"path": path, "name": Path(path).name, "duration": probe(path)}


@app.post("/api/probe")
def probe_path(body: Relink):
    path = os.path.expanduser((body.path or "").strip().strip("'\""))
    if not os.path.isfile(path):
        raise HTTPException(400, "Файл не найден.")
    return {"path": path, "name": Path(path).name, "duration": probe(path)}


@app.post("/api/calls")
def create_call(body: NewCall):
    path = os.path.expanduser(body.path)
    if not os.path.isfile(path):
        raise HTTPException(400, "Файл не найден.")
    duration = probe(path)
    title, started = nice_title(path)
    with closing(db()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO calls (title, path, duration, recorded_at, created_at, speakers, merge, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')",
            (title, path, duration, started, time.time(),
             body.speakers, int(body.merge)))
        call_id = cur.lastrowid
    jobs.put(call_id)
    return {"id": call_id}


@app.get("/api/calls")
def list_calls():
    with closing(db()) as conn:
        rows = conn.execute("SELECT id, title, duration, recorded_at, status, stage, progress "
                            "FROM calls ORDER BY recorded_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/calls/{call_id}")
def get_call(call_id: int):
    with closing(db()) as conn:
        call = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
        if call is None:
            raise HTTPException(404, "Звонок не найден.")
        speakers = conn.execute("SELECT label, name, idx FROM speakers WHERE call_id=? "
                                "ORDER BY idx", (call_id,)).fetchall()
        segments = conn.execute("SELECT id, start, end, speaker, text FROM segments "
                                "WHERE call_id=? ORDER BY start", (call_id,)).fetchall()
    result = dict(call)
    result["media_exists"] = os.path.isfile(call["path"])
    result["has_video"] = has_video(call["path"]) if result["media_exists"] else False
    result["speakers"] = [dict(s) for s in speakers]
    result["segments"] = [dict(s) for s in segments]
    return result


def has_video(path: str) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=codec_type", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    return "video" in out.stdout


@app.get("/api/calls/{call_id}/media")
def media(call_id: int):
    with closing(db()) as conn:
        call = conn.execute("SELECT path FROM calls WHERE id=?", (call_id,)).fetchone()
    if call is None or not os.path.isfile(call["path"]):
        raise HTTPException(404, "Запись не найдена.")
    return FileResponse(call["path"])


@app.post("/api/calls/{call_id}/relink")
def relink(call_id: int, body: Relink):
    path = os.path.expanduser(body.path) if body.path else choose_file("Где теперь эта запись?")
    if not os.path.isfile(path):
        raise HTTPException(400, "Файл не найден.")
    set_status(call_id, path=path)
    return {"path": path}


@app.post("/api/calls/{call_id}/cancel")
def cancel(call_id: int):
    with closing(db()) as conn:
        call = conn.execute("SELECT status FROM calls WHERE id=?", (call_id,)).fetchone()
    if call and call["status"] == "queued":
        set_status(call_id, status="cancelled")
    elif call and call["status"] == "running":
        cancelled.add(call_id)
    return {"ok": True}


@app.post("/api/calls/{call_id}/retry")
def retry(call_id: int):
    set_status(call_id, status="queued", stage=None, progress=0, error=None)
    jobs.put(call_id)
    return {"ok": True}


@app.delete("/api/calls/{call_id}")
def delete_call(call_id: int):
    cancelled.add(call_id)
    with closing(db()) as conn, conn:
        conn.execute("DELETE FROM calls WHERE id=?", (call_id,))
    return {"ok": True}


@app.patch("/api/calls/{call_id}")
def rename_call(call_id: int, body: Rename):
    title = body.name.strip()
    if not title:
        raise HTTPException(400, "Название не может быть пустым.")
    with closing(db()) as conn, conn:
        conn.execute("UPDATE calls SET title=? WHERE id=?", (title, call_id))
    return {"ok": True}


@app.patch("/api/calls/{call_id}/speakers/{label}")
def rename_speaker(call_id: int, label: str, body: Rename):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Имя не может быть пустым.")
    with closing(db()) as conn, conn:
        conn.execute("UPDATE speakers SET name=? WHERE call_id=? AND label=?",
                     (name, call_id, label))
    return {"ok": True}


@app.patch("/api/segments/{segment_id}")
def edit_segment(segment_id: int, body: EditText):
    with closing(db()) as conn, conn:
        conn.execute("UPDATE segments SET text=? WHERE id=?", (body.text.strip(), segment_id))
    return {"ok": True}


@app.get("/api/search")
def search(q: str):
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT s.id, s.call_id, s.start, s.text, c.title, sp.name AS speaker "
            "FROM segments s JOIN calls c ON c.id = s.call_id "
            "JOIN speakers sp ON sp.call_id = s.call_id AND sp.label = s.speaker "
            "WHERE s.text LIKE ? ORDER BY c.recorded_at DESC, s.start LIMIT 200",
            (f"%{q}%",)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/calls/{call_id}/export.{fmt}")
def export(call_id: int, fmt: str):
    call = get_call(call_id)
    names = {s["label"]: s["name"] for s in call["speakers"]}
    lines = [f"[{c.fmt(s['start'])}–{c.fmt(s['end'])}] {names[s['speaker']]}: {s['text']}"
             for s in call["segments"]]
    if fmt == "md":
        body = f"# {call['title']}\n\n" + "\n\n".join(
            f"**{names[s['speaker']]}** · {c.fmt(s['start'])}  \n{s['text']}"
            for s in call["segments"]) + "\n"
    elif fmt == "txt":
        body = "\n".join(lines) + "\n"
    else:
        raise HTTPException(404)
    filename = f"{call['title']}.{fmt}"
    return PlainTextResponse(body, headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"})


def quote(s: str) -> str:
    from urllib.parse import quote as q
    return q(s)


def main():
    import uvicorn
    init_db()
    threading.Thread(target=worker, daemon=True).start()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
