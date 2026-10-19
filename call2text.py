"""Расшифровка звонков: pyannote (кто говорит) + GigaAM-v3 e2e RNNT (что сказал).

Использование: call2text запись.webm [--speakers N] [--no-merge] [-o результат.txt]
  --speakers  сколько собеседников на записи (если известно — точнее разметка)
  --no-merge  не склеивать соседние реплики одного собеседника (по умолчанию склеиваются при паузе < 1 с)
  -o          куда сохранить текст (по умолчанию .txt рядом с записью)
Нужен HF_TOKEN с доступом к pyannote/segmentation-3.0 и speaker-diarization-community-1.
"""

import argparse
import os
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from transformers import AutoModel

# torch >= 2.6 по умолчанию не грузит чекпоинты pyannote без этого списка
torch.serialization.add_safe_globals(
    [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
)

SR = 16000
SHORT_LIMIT = 25.0  # больше — только transcribe_longform
MERGE_GAP = 1.0
MIN_TURN = 0.2  # обрывки короче модель не принимает, слов в них нет


def load_audio(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn",
           "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def load_pipeline():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=os.environ["HF_TOKEN"]
    )
    if torch.cuda.is_available():  # видеокарта NVIDIA (Windows, Linux)
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():  # видеокарта Mac, в ~6 раз быстрее
        pipeline.to(torch.device("mps"))
    return pipeline


def load_model():
    return AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision="e2e_rnnt",
                                     trust_remote_code=True)


def diarize(wav: np.ndarray, speakers=None, pipeline=None, hook=None) -> list:
    pipeline = pipeline or load_pipeline()
    output = pipeline({"waveform": torch.from_numpy(wav)[None], "sample_rate": SR},
                      num_speakers=speakers, hook=hook)
    return [(turn.start, turn.end, speaker)
            for turn, speaker in output.exclusive_speaker_diarization]


def merge_turns(turns: list) -> list:
    merged = []
    for start, end, speaker in turns:
        if merged and merged[-1][2] == speaker and start - merged[-1][1] < MERGE_GAP:
            merged[-1] = (merged[-1][0], end, speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def transcribe_turn(model, wav: np.ndarray, start: float, end: float, chunk_path: str) -> str:
    sf.write(chunk_path, wav[int(start * SR):int(end * SR)], SR)
    if end - start <= SHORT_LIMIT:
        return model.transcribe(chunk_path).strip()
    return " ".join(seg["transcription"]
                    for seg in model.transcribe_longform(chunk_path)).strip()


def transcribe_turns(model, wav: np.ndarray, turns: list) -> list:
    names, lines = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = os.path.join(tmp, "turn.wav")
        for start, end, speaker in turns:
            if end - start < MIN_TURN:
                continue
            text = transcribe_turn(model, wav, start, end, chunk_path)
            if not text:
                continue
            name = names.setdefault(speaker, f"Собеседник {len(names) + 1}")
            lines.append(f"[{fmt(start)}–{fmt(end)}] {name}: {text}")
    return lines


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    wav = load_audio(args.audio)
    turns = diarize(wav, args.speakers)
    if not args.no_merge:
        turns = merge_turns(turns)

    model = load_model()
    lines = transcribe_turns(model, wav, turns)

    output = args.output or os.path.splitext(args.audio)[0] + ".txt"
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {output}")


if __name__ == "__main__":
    main()
