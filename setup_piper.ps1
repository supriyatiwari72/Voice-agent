# setup_piper.ps1
# Automates the download and installation of Piper TTS and its voice model files.

$ErrorActionPreference = "Stop"

Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path .\piper | Out-Null
New-Item -ItemType Directory -Force -Path .\weights | Out-Null

Write-Host "Downloading Piper for Windows (AMD64)..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip" -OutFile .\piper\piper.zip

Write-Host "Extracting archive..." -ForegroundColor Cyan
Expand-Archive -Path .\piper\piper.zip -DestinationPath .\piper\extracted -Force

# Handle folder nesting inside the zip
if (Test-Path .\piper\extracted\piper\piper.exe) {
    Move-Item -Path .\piper\extracted\piper\* -Destination .\piper -Force
} else {
    Move-Item -Path .\piper\extracted\* -Destination .\piper -Force
}

Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
Remove-Item -Path .\piper\extracted -Recurse -Force
Remove-Item -Path .\piper\piper.zip -Force

Write-Host "Downloading en_US-lessac-medium.onnx model..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -OutFile .\weights\en_US-lessac-medium.onnx

Write-Host "Downloading en_US-lessac-medium.onnx.json configuration..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -OutFile .\weights\en_US-lessac-medium.onnx.json

Write-Host "Piper setup complete successfully!" -ForegroundColor Green
