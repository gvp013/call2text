# Установка call2text на Windows (экспериментально):
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/gvp013/call2text/main/install.ps1 | iex"
# Или из скачанной папки проекта: powershell -ExecutionPolicy Bypass -File install.ps1
# Повторный запуск безопасен: обновит проект и зависимости, токен спросит только если он не работает.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # иначе Invoke-WebRequest качает в разы медленнее
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Repo = if ($env:CALL2TEXT_REPO) { $env:CALL2TEXT_REPO } else { "https://github.com/gvp013/call2text" }
$Dir = if ($env:CALL2TEXT_DIR) { $env:CALL2TEXT_DIR } else { Join-Path $HOME "call2text" }
$Models = "pyannote/segmentation-3.0", "pyannote/speaker-diarization-community-1"

function Say($text) { Write-Host "`n$text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`n$text" -ForegroundColor Red; exit 1 }
function Refresh-Path {
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
              [Environment]::GetEnvironmentVariable("Path", "User") + ";" + (Join-Path $HOME ".local\bin")
}

# --- 1. Windows 10/11 на x64 ---
if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") { Fail "Нужна 64-битная Windows на процессоре Intel или AMD." }
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  Fail "Нет winget. Установите «Установщик приложений» (App Installer) из Microsoft Store и запустите установку ещё раз."
}
function Winget($id) {
  winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
  Refresh-Path
}

# --- 2. Библиотеки Visual C++ (нужны PyTorch) и FFmpeg ---
if (-not (Test-Path "$env:SystemRoot\System32\vcruntime140_1.dll")) {
  Say "Ставлю библиотеки Visual C++…"
  Winget "Microsoft.VCRedist.2015+.x64"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Say "Ставлю FFmpeg…"
  Winget "Gyan.FFmpeg"
  if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "FFmpeg не установился. Перезапустите PowerShell и запустите установку ещё раз." }
}

# --- 3. uv (сам скачает Python 3.12) ---
Refresh-Path
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say "Ставлю uv…"
  $env:UV_NO_MODIFY_PATH = "1"
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  Refresh-Path
}

# --- 4. Проект ---
if ($PSCommandPath -and (Test-Path (Join-Path (Split-Path $PSCommandPath) "call2text.py"))) {
  $Dir = Split-Path $PSCommandPath  # запущен из папки проекта — ставим прямо в неё
} else {
  Say "Скачиваю проект в $Dir…"
  $zip = Join-Path $env:TEMP "call2text.zip"
  $tmp = Join-Path $env:TEMP "call2text-unpack"
  Invoke-WebRequest "$Repo/archive/refs/heads/main.zip" -OutFile $zip
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive $zip $tmp
  New-Item -ItemType Directory -Force $Dir | Out-Null
  # .venv и data (база с расшифровками) при обновлении не трогаем
  Copy-Item (Join-Path (Get-ChildItem $tmp)[0].FullName "*") $Dir -Recurse -Force
  Remove-Item $zip, $tmp -Recurse -Force
}

Say "Ставлю Python 3.12 и библиотеки (первый раз — несколько минут, 2–5 ГБ)…"
$Python = Join-Path $Dir ".venv\Scripts\python.exe"
uv venv --quiet --allow-existing --python 3.12 (Join-Path $Dir ".venv")
if ($LASTEXITCODE) { Fail "Не удалось создать окружение Python." }
# Сначала PyTorch под видеокарту NVIDIA, если она есть (иначе версия для процессора), потом остальное
# с PyPI: у torchcodec нет CUDA-сборки под Windows
uv pip install --quiet --python $Python --torch-backend auto torch==2.10.0 torchaudio==2.10.0
if ($LASTEXITCODE) { Fail "Не удалось установить PyTorch." }
uv pip install --quiet --python $Python -r (Join-Path $Dir "requirements.txt")
if ($LASTEXITCODE) { Fail "Не удалось установить библиотеки." }

# --- 5. Команды и ярлык ---
$env_lines = "@echo off`r`nset PYTHONUTF8=1`r`nset HF_HUB_DISABLE_SYMLINKS_WARNING=1`r`n"
Set-Content (Join-Path $Dir "calls-ui.bat") ($env_lines + "`"%~dp0.venv\Scripts\python.exe`" `"%~dp0ui\server.py`" %*`r`n") -Encoding ASCII
Set-Content (Join-Path $Dir "call2text.bat") ($env_lines + "`"%~dp0.venv\Scripts\python.exe`" `"%~dp0call2text.py`" %*`r`n") -Encoding ASCII

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if (($userPath -split ";") -notcontains $Dir) {
  [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $Dir).TrimStart(";"), "User")
}

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Расшифровка звонков.lnk"))
$link.TargetPath = Join-Path $Dir "calls-ui.bat"
$link.WorkingDirectory = $Dir
$link.Save()

# --- 6. Токен HuggingFace ---
function Test-Token($token) {
  $headers = @{ Authorization = "Bearer $token" }
  try { Invoke-WebRequest https://huggingface.co/api/whoami-v2 -Headers $headers -UseBasicParsing | Out-Null }
  catch { Write-Host "Токен не принят HuggingFace."; return $false }
  foreach ($m in $Models) {
    try { Invoke-WebRequest "https://huggingface.co/$m/resolve/main/config.yaml" -Headers $headers -Method Head -UseBasicParsing | Out-Null }
    catch { Write-Host "Нет доступа к $m — примите условия на https://huggingface.co/$m"; return $false }
  }
  return $true
}

$Token = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
if ($Token -and (Test-Token $Token)) {
  Say "Токен HuggingFace уже настроен."
} else {
  Say "Нужен бесплатный токен HuggingFace — им скачивается модель разметки собеседников."
  Write-Host "1. Войдите или зарегистрируйтесь на huggingface.co."
  Write-Host "2. На двух страницах моделей, которые сейчас откроются, заполните форму и примите условия."
  Write-Host "3. Создайте токен с правом Read на странице токенов."
  Read-Host "Нажмите Enter, чтобы открыть страницы" | Out-Null
  foreach ($m in $Models) { Start-Process "https://huggingface.co/$m" }
  Start-Process "https://huggingface.co/settings/tokens"
  while ($true) {
    $secure = Read-Host "Вставьте токен (hf_…, ввод не отображается)" -AsSecureString
    $Token = ([Net.NetworkCredential]::new("", $secure).Password).Trim()
    if (Test-Token $Token) { break }
    Write-Host "Проверьте, что токен скопирован целиком и условия обеих моделей приняты."
  }
  [Environment]::SetEnvironmentVariable("HF_TOKEN", $Token, "User")
  Say "Токен работает и сохранён в переменных среды пользователя."
}
$env:HF_TOKEN = $Token

# --- 7. Модели заранее ---
$answer = Read-Host "Скачать модели сейчас (~1,5 ГБ), чтобы первая расшифровка не ждала? [Д/н]"
if ($answer -notmatch "^[нНnN]") {
  Say "Скачиваю модели…"
  $env:PYTHONUTF8 = "1"
  Push-Location $Dir
  & $Python -c "import call2text as c; c.load_pipeline(); c.load_model()"
  $code = $LASTEXITCODE
  Pop-Location
  if ($code) { Fail "Модели не скачались. Запустите установку ещё раз — скачанное не пропадёт." }
}

$gpu = & $Python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
Say "Готово."
if ($gpu) { Write-Host "Видеокарта: $gpu — разметка пойдёт на ней." }
else { Write-Host "Видеокарта NVIDIA не найдена — всё будет работать на процессоре, медленнее (15 минут звонка ≈ 10–15 минут)." }
Write-Host "Интерфейс: ярлык «Расшифровка звонков» на рабочем столе."
Write-Host "Консоль:   call2text запись.webm --speakers 2  (в новом окне терминала)"
