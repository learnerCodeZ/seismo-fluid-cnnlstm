@echo off
chcp 65001 >nul
echo ========================================
echo   地震流体异常检测可视化平台 - 启动
echo ========================================
echo.

cd /d "%~dp0\.."

set "PYTHON=D:\桌面\地质\大创\dataAnalysis\.venv\Scripts\python.exe"
set PORT=8002

echo [1/2] 检查依赖...
"%PYTHON%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo 安装 FastAPI 和 uvicorn...
    "%PYTHON%" -m pip install fastapi uvicorn --quiet
)

echo [2/2] 启动服务...
echo.

REM 检查端口是否被占用
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 端口 %PORT% 已被占用，服务可能已在运行。
    echo.
    echo 直接访问: http://localhost:%PORT%
    echo.
    pause
    exit /b 0
)

echo 访问地址: http://localhost:%PORT%
echo 按 Ctrl+C 停止服务
echo.
"%PYTHON%" -m uvicorn web.app:app --host 127.0.0.1 --port %PORT%
pause
