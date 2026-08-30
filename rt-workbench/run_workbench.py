#!/usr/bin/env python3
"""RT 판독 워크벤치 실행 런처 — 터미널 명령 없이 파일 하나로 실행.

사용법 (아무거나):
  - Windows: 같은 폴더의 `실행.bat` 더블클릭 (또는 이 파일을 더블클릭 — .py가
    Python에 연결되어 있을 때)
  - macOS/Linux: `./run_workbench.sh` 또는 `python3 run_workbench.py`

하는 일:
  1) 필수 패키지 확인 — 없으면 설치 여부를 물어보고 자동 설치
  2) 샘플 필름이 없으면 자동 생성 (scripts/generate_samples.py)
  3) 빈 포트를 찾아 Streamlit 서버 기동 + 기본 브라우저 자동 오픈
종료: 이 창에서 Ctrl+C.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (import 이름, pip 패키지 이름)
_REQUIRED = [
    ("streamlit", "streamlit"),
    ("streamlit_image_coordinates", "streamlit-image-coordinates"),
    ("cv2", "opencv-python-headless"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("pandas", "pandas"),
    ("reportlab", "reportlab"),
]


def check_deps() -> None:
    """필수 패키지 확인. 없으면 (대화형일 때) 설치를 제안한다."""
    missing = []
    for mod, pkg in _REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return

    print("필요한 패키지가 설치되어 있지 않습니다:", ", ".join(missing))
    answer = ""
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            answer = input("지금 자동으로 설치할까요? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
    if answer == "y":
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        return
    print("\n다음 명령으로 설치한 뒤 다시 실행해 주세요:")
    print(f"  {Path(sys.executable).name} -m pip install -r {ROOT / 'requirements.txt'}")
    sys.exit(1)


def ensure_samples() -> None:
    """샘플 필름이 없으면 생성한다 (처음 실행 시 1회)."""
    samples = ROOT / "data" / "samples"
    if samples.exists() and any(samples.glob("*.png")):
        return
    print("샘플 필름이 없어 생성합니다... (합성 RT 이미지 6장)")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "generate_samples.py")])


def find_free_port(start: int = 8501, tries: int = 50) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main() -> None:
    check_deps()
    ensure_samples()

    port = find_free_port()
    print(f"\nRT 판독 워크벤치를 시작합니다 → http://localhost:{port}")
    print("브라우저 창이 자동으로 열립니다. 종료하려면 이 창에서 Ctrl+C.\n")

    # streamlit CLI를 프로세스 내에서 직접 구동 — 사용자는 명령을 칠 필요가 없다.
    # (headless가 아니므로 streamlit이 기본 브라우저를 자동으로 연다.)
    sys.argv = [
        "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    from streamlit.web import cli as stcli

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
