@echo off
rem RT 판독 워크벤치 — Windows 더블클릭 실행용
chcp 65001 > nul
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python run_workbench.py
    goto :done
)
where py >nul 2>nul
if %errorlevel%==0 (
    py run_workbench.py
    goto :done
)
echo Python이 설치되어 있지 않습니다.
echo https://www.python.org/downloads/ 에서 설치(Add to PATH 체크) 후 다시 실행해 주세요.

:done
pause
