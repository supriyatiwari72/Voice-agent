#!/usr/bin/env python3
"""
setup_project.py — single-command setup for the Voice Agent project.

Usage:
    python setup_project.py

What it does:
    1.  Checks Python version (3.10+ required)
    2.  Creates a virtual environment (if not already in one)
    3.  Installs all Python packages (requirements.txt + optional extras)
    4.  Installs system dependencies (PortAudio, etc.) via sudo
    5.  Creates .env from .env.example if missing
    6.  Downloads Kokoro TTS model weights
    7.  Verifies key imports work
    8.  Shows next steps (Ollama setup, running the agent)
"""

import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile

# ──────────────────────────────────────────────────────────────────────────────
#  Colour helpers
# ──────────────────────────────────────────────────────────────────────────────

class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {C.GREEN}[✓]{C.RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {C.CYAN}[i]{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}[!]{C.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}[✗]{C.RESET} {msg}")
    sys.exit(1)


def section(title: str) -> None:
    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def in_venv() -> bool:
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def detect_package_manager() -> str:
    for pm in ["apt-get", "dnf", "yum", "pacman", "zypper", "brew", "pkg"]:
        if shutil.which(pm):
            return pm
    return ""


# ──────────────────────────────────────────────────────────────────────────────
#  Steps
# ──────────────────────────────────────────────────────────────────────────────

def step_check_python() -> None:
    section("1. Checking Python version")

    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        fail(f"Python 3.10+ required, found {v.major}.{v.minor}.{v.micro}")
    ok(f"Python {v.major}.{v.minor}.{v.micro}")


def step_venv() -> None:
    section("2. Virtual environment")

    if in_venv():
        ok("Already inside a virtual environment")
        return

    venv_path = os.path.join(PROJECT_DIR, "venv")
    if os.path.isdir(venv_path):
        info(f"Found existing venv at {venv_path}")
    else:
        info("Creating virtual environment…")
        run([sys.executable, "-m", "venv", venv_path])
        ok("Virtual environment created")

    pip_path = os.path.join(venv_path, "bin", "pip")
    info(f"To activate:  source {venv_path}/bin/activate")
    info(f"Then re-run:  python setup_project.py")
    ok("Virtual environment ready")


def step_install_python_packages() -> None:
    section("3. Installing Python packages")

    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    if not os.path.isfile(req_file):
        warn("requirements.txt not found — skipping pip install")
        return

    python = sys.executable

    info("Installing core dependencies from requirements.txt…")
    result = run([python, "-m", "pip", "install", "-r", req_file], check=False)
    if result.returncode != 0:
        warn(f"pip install had warnings/errors:\n{result.stderr}")
    else:
        ok("Core dependencies installed")

    info("Installing optional extras (noisereduce, websocket-client)…")
    extras_result = run(
        [python, "-m", "pip", "install", "noisereduce", "websocket-client"],
        check=False,
    )
    if extras_result.returncode == 0:
        ok("Optional extras installed")
    else:
        warn("Optional extras had issues (noisereduce/websocket-client) — will be skipped at runtime if missing")

    ok("Python packages ready")


def step_system_deps() -> None:
    section("4. System dependencies")

    pm = detect_package_manager()
    if not pm:
        warn("No supported package manager found — install PortAudio manually")
        return

    if pm in ("apt-get",):
        packages = ["portaudio19-dev", "python3-pip"]
    elif pm in ("dnf", "yum"):
        packages = ["portaudio-devel", "alsa-lib-devel", "python3-pip"]
    elif pm == "pacman":
        packages = ["portaudio", "python-pip"]
    elif pm == "brew":
        packages = ["portaudio"]
    else:
        warn(f"Unknown package manager '{pm}' — install PortAudio manually")
        return

    info(f"Detected package manager: {pm}")
    info(f"Packages to install: {' '.join(packages)}")

    need_sudo = pm not in ("brew", "pkg")
    cmd = (["sudo"] if need_sudo else []) + [pm, "install", "-y"] + packages

    result = run(cmd, check=False)
    if result.returncode != 0:
        warn(f"System package install had issues:\n{result.stderr}")
    else:
        ok("System dependencies installed")


def step_env_file() -> None:
    section("5. Environment file")

    env_path = os.path.join(PROJECT_DIR, ".env")
    example_path = os.path.join(PROJECT_DIR, ".env.example")

    if os.path.isfile(env_path):
        ok(".env already exists — skipping")
        return

    if not os.path.isfile(example_path):
        warn(".env.example not found — create .env manually")
        return

    shutil.copy2(example_path, env_path)
    ok(".env created from .env.example — edit it to add your API keys")


def step_download_weights() -> None:
    section("6. Kokoro TTS model weights")

    weights_dir = os.path.join(PROJECT_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    files_needed = {
        "kokoro-v0_19.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "voices.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin",
    }

    all_present = all(os.path.isfile(os.path.join(weights_dir, f)) for f in files_needed)
    if all_present:
        ok("All Kokoro model files present")
        return

    missing = [f for f in files_needed if not os.path.isfile(os.path.join(weights_dir, f))]
    warn(f"Missing files: {', '.join(missing)}")
    info("Downloading Kokoro TTS model weights from HuggingFace…")

    for filename, url in files_needed.items():
        dest = os.path.join(weights_dir, filename)
        if os.path.isfile(dest):
            info(f"  {filename} already exists — skipping")
            continue
        info(f"  Downloading {filename} (this may take a while)…")
        try:
            urllib.request.urlretrieve(url, dest)
            ok(f"  {filename} downloaded")
        except Exception as e:
            warn(f"  Failed to download {filename}: {e}")
            warn("  Download manually from: https://github.com/thewh1teagle/kokoro-onnx/releases")

    ok("Kokoro weights ready")


def step_verify_imports() -> None:
    section("7. Verifying installation")

    checks = [
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("requests", "requests"),
        ("sounddevice", "sounddevice"),
        ("scipy.signal", "scipy"),
        ("dotenv", "python-dotenv"),
    ]

    all_good = True
    for modname, pkgname in checks:
        try:
            __import__(modname)
            ok(f"{pkgname:<20s} found")
        except ImportError:
            warn(f"{pkgname:<20s} NOT found")
            all_good = False

    try:
        import noisereduce
        ok("noisereduce            found (optional)")
    except ImportError:
        info("noisereduce            not installed (optional — noise reduction via noisereduce disabled)")

    try:
        import websocket
        ok("websocket-client       found (optional)")
    except ImportError:
        info("websocket-client       not installed (optional — cloud STT providers only)")

    try:
        import torch
        ok("torch                 found")
    except ImportError:
        warn("torch                 NOT found — Silero VAD will not work")
        warn("  Install with: pip install torch")
        all_good = False

    try:
        import faster_whisper
        ok("faster-whisper         found")
    except ImportError:
        warn("faster-whisper         NOT found — DistilWhisper / FasterWhisper STT will not work")
        all_good = False

    try:
        import kokoro_onnx
        ok("kokoro-onnx            found")
    except ImportError:
        warn("kokoro-onnx            NOT found — Kokoro TTS will not work")
        all_good = False

    if all_good:
        ok("All core dependencies verified")
    else:
        warn("Some core dependencies are missing — check warnings above")


def step_next_steps() -> None:
    section("8. Next steps")

    print(f"""
  {C.BOLD}Setup is complete!{C.RESET}

  {C.CYAN}1. Activate the virtual environment (if not already):{C.RESET}
       source {os.path.join(PROJECT_DIR, "venv")}/bin/activate

  {C.CYAN}2. Set up Ollama for local LLM (recommended):{C.RESET}
       curl -fsSL https://ollama.ai/install.sh | sh
       ollama serve
       ollama pull qwen2.5:3b

  {C.CYAN}3. Edit .env with your API keys (optional):{C.RESET}
       nano {os.path.join(PROJECT_DIR, ".env")}
       {C.YELLOW}At minimum set GROQ_API_KEY for cloud LLM.{C.RESET}

  {C.CYAN}4. Run the agent:{C.RESET}
       python voice_agent.py

  {C.CYAN}5. Run tests:{C.RESET}
       python -m pytest tests/ -v

  {C.CYAN}6. List available providers:{C.RESET}
       python voice_agent.py --list
""")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"""
    {C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗
    ║     Voice Agent — Project Setup Script     ║
    ╚══════════════════════════════════════════════════╝{C.RESET}
""")

    step_check_python()
    step_venv()

    if not in_venv() and os.path.isdir(os.path.join(PROJECT_DIR, "venv")):
        venv_python = os.path.join(PROJECT_DIR, "venv", "bin", "python")
        if os.path.isfile(venv_python):
            info("Re-running setup inside virtual environment…")
            os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])

    step_install_python_packages()
    step_system_deps()
    step_env_file()
    step_download_weights()
    step_verify_imports()
    step_next_steps()


if __name__ == "__main__":
    main()
