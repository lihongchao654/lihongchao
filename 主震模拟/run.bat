@echo off
REM ============================================================================
REM 地震模拟系统 - 快速启动脚本（批处理）
REM ============================================================================

chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  地震模拟系统 - 纯 Python 版本启动
echo ============================================================================
echo.

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM 检查虚拟环境
if exist ".venv\Scripts\python.exe" (
    echo ✓ 虚拟环境已存在
) else (
    echo ✗ 虚拟环境不存在，请先运行 setup_env_with_python311.ps1
    echo   或在 PowerShell 中执行：
    echo   python -m venv .venv
    echo   .\.venv\Scripts\pip install numpy pandas scipy matplotlib
    pause
    exit /b 1
)

REM 激活虚拟环境
echo.
echo [1] 激活虚拟环境...
call .venv\Scripts\activate.bat
echo ✓ 虚拟环境已激活

REM 检查脚本文件
if not exist "人工模拟地震动_纯Python版.py" (
    echo ✗ 找不到 人工模拟地震动_纯Python版.py
    pause
    exit /b 1
)

REM 运行脚本
echo.
echo [2] 启动地震模拟系统...
echo ============================================================================
echo.
python "人工模拟地震动_纯Python版.py"

echo.
echo ============================================================================
echo ✓ 模拟完成！按任意键退出...
echo ============================================================================
pause

echo.
echo ============================================================================
echo 使用说明：
echo  1. 请确保已安装 Python 3.11 及以上版本
echo  2. 请确保已安装必要的依赖库（见上方虚拟环境检查）
echo  3. 双击项目目录中的 run.bat 文件
echo ============================================================================
pause
