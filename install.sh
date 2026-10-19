#!/bin/bash
# Установка call2text на Mac:
#   curl -fsSL https://raw.githubusercontent.com/gvp013/call2text/main/install.sh | bash
# Или из скачанной папки проекта: ./install.sh
# Повторный запуск безопасен: обновит проект и зависимости, токен спросит только если он не работает.
set -euo pipefail

REPO="${CALL2TEXT_REPO:-https://github.com/gvp013/call2text.git}"
DIR="${CALL2TEXT_DIR:-$HOME/gigaam-hf}"
BIN="$HOME/.local/bin"
PROFILE="$HOME/.zprofile"
MODELS=(pyannote/segmentation-3.0 pyannote/speaker-diarization-community-1)

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
ask()  { { read -r "$@" </dev/tty; } 2>/dev/null || read -r "$@"; }  # при curl | bash stdin занят скриптом

# --- 1. Mac на Apple Silicon ---
[ "$(uname -s)" = Darwin ] || fail "Программа сделана только для macOS."
[ "$(uname -m)" = arm64 ] || fail "Нужен Mac на Apple Silicon (M1 и новее)."

# --- 2. FFmpeg ---
if ! command -v ffmpeg >/dev/null; then
  command -v brew >/dev/null || fail "Нужен Homebrew: поставьте его с https://brew.sh и запустите установку ещё раз."
  say "Ставлю FFmpeg…"
  brew install ffmpeg
fi

# --- 3. uv (сам скачает Python 3.12) ---
export PATH="$BIN:$PATH"
if ! command -v uv >/dev/null; then
  say "Ставлю uv…"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

# --- 4. Проект ---
SELF_DIR=""
[ -f "${BASH_SOURCE[0]:-}" ] && SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$SELF_DIR" ] && [ -f "$SELF_DIR/call2text.py" ]; then
  DIR="$SELF_DIR"  # запущен из папки проекта — ставим прямо в неё
elif [ -d "$DIR/.git" ]; then
  say "Обновляю проект в $DIR…"
  git -C "$DIR" pull --ff-only
else
  say "Скачиваю проект в $DIR…"
  git clone --depth 1 "$REPO" "$DIR"
fi

say "Ставлю Python 3.12 и библиотеки (первый раз — несколько минут, ~2 ГБ)…"
uv venv --quiet --allow-existing --python 3.12 "$DIR/.venv"
uv pip install --quiet --python "$DIR/.venv/bin/python" -r "$DIR/requirements.txt"

# --- 5. Команды и ярлык ---
mkdir -p "$BIN"
for pair in "call2text:call2text.py" "calls-ui:ui/server.py"; do
  cat > "$BIN/${pair%%:*}" <<EOF
#!/bin/sh
. "\$HOME/.zprofile" 2>/dev/null
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$DIR/.venv/bin/python" "$DIR/${pair#*:}" "\$@"
EOF
  chmod +x "$BIN/${pair%%:*}"
done

touch "$PROFILE"
grep -q '.local/bin' "$PROFILE" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE"

SHORTCUT="$HOME/Desktop/Расшифровка звонков.command"
printf '#!/bin/sh\nexec "%s/calls-ui"\n' "$BIN" > "$SHORTCUT"
chmod +x "$SHORTCUT"

# --- 6. Токен HuggingFace ---
token_ok() {
  curl -fs -o /dev/null -H "Authorization: Bearer $1" https://huggingface.co/api/whoami-v2 || return 1
  for m in "${MODELS[@]}"; do
    curl -fs -o /dev/null -H "Authorization: Bearer $1" "https://huggingface.co/$m/resolve/main/config.yaml" \
      || { echo "Нет доступа к $m — примите условия на https://huggingface.co/$m"; return 1; }
  done
}

TOKEN="$( . "$PROFILE" >/dev/null 2>&1; echo "${HF_TOKEN:-}")"
if [ -n "$TOKEN" ] && token_ok "$TOKEN" >/dev/null 2>&1; then
  say "Токен HuggingFace уже настроен."
else
  say "Нужен бесплатный токен HuggingFace — им скачивается модель разметки собеседников."
  echo "1. Войдите или зарегистрируйтесь на huggingface.co."
  echo "2. На двух страницах моделей, которые сейчас откроются, заполните форму и примите условия."
  echo "3. Создайте токен с правом Read на странице токенов."
  ask -p "Нажмите Enter, чтобы открыть страницы… "
  for m in "${MODELS[@]}"; do open "https://huggingface.co/$m"; done
  open "https://huggingface.co/settings/tokens"
  while :; do
    ask -s -p "Вставьте токен (hf_…, ввод не отображается): " TOKEN; echo
    TOKEN="$(printf '%s' "$TOKEN" | tr -d '[:space:]')"
    token_ok "$TOKEN" && break
    echo "Токен не подошёл. Проверьте, что он скопирован целиком и условия обеих моделей приняты."
  done
  sed -i '' '/^export HF_TOKEN=/d' "$PROFILE"
  echo "export HF_TOKEN=$TOKEN" >> "$PROFILE"
  say "Токен работает и сохранён в $PROFILE."
fi

# --- 7. Модели заранее ---
ask -p "Скачать модели сейчас (~1,5 ГБ), чтобы первая расшифровка не ждала? [Д/н] " ANSWER
case "${ANSWER:-д}" in
  н*|Н*|n*|N*) ;;
  *) say "Скачиваю модели…"
     (cd "$DIR" && HF_TOKEN="$TOKEN" PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -c \
       "import call2text as c; c.load_pipeline(); c.load_model()") \
       || fail "Модели не скачались. Запустите установку ещё раз — скачанное не пропадёт." ;;
esac

say "Готово."
echo "Интерфейс: ярлык «Расшифровка звонков» на рабочем столе или команда calls-ui."
echo "Консоль:   call2text запись.webm --speakers 2"
echo "Откройте новое окно терминала, чтобы команды стали доступны."
