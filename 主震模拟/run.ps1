# ============================================================================
# 地震模拟系统 - 快速启动脚本（PowerShell）
# ============================================================================

# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# 定义颜色
$colors = @{
    'success' = 'Green'
    'error'   = 'Red'
    'info'    = 'Cyan'
    'warn'    = 'Yellow'
}

# 获取脚本所在目录
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "============================================================================" -ForegroundColor $colors['info']
Write-Host "  地震模拟系统 - 纯 Python 版本启动" -ForegroundColor $colors['info']
Write-Host "============================================================================" -ForegroundColor $colors['info']
Write-Host ""

# 检查虚拟环境
$venvPath = Join-Path $scriptDir ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (Test-Path $pythonExe) {
    Write-Host "✓ 虚拟环境已存在" -ForegroundColor $colors['success']
} else {
    Write-Host "✗ 虚拟环境不存在" -ForegroundColor $colors['error']
    Write-Host "  请先运行：python -m venv .venv" -ForegroundColor $colors['warn']
    Write-Host "  然后运行：.\.venv\Scripts\pip install numpy pandas scipy matplotlib" -ForegroundColor $colors['warn']
    exit 1
}

# 激活虚拟环境
Write-Host ""
Write-Host "[1] 激活虚拟环境..." -ForegroundColor $colors['info']
& (Join-Path $venvPath "Scripts\Activate.ps1")
Write-Host "✓ 虚拟环境已激活" -ForegroundColor $colors['success']

# 检查脚本文件
$scriptFile = Join-Path $scriptDir "人工模拟地震动_纯Python版.py"
if (-not (Test-Path $scriptFile)) {
    Write-Host "✗ 找不到 人工模拟地震动_纯Python版.py" -ForegroundColor $colors['error']
    exit 1
}

# 运行脚本
Write-Host ""
Write-Host "[2] 启动地震模拟系统..." -ForegroundColor $colors['info']
Write-Host "============================================================================" -ForegroundColor $colors['info']
Write-Host ""

& python $scriptFile

Write-Host ""
Write-Host "============================================================================" -ForegroundColor $colors['info']
Write-Host "✓ 模拟完成！" -ForegroundColor $colors['success']
Write-Host "============================================================================" -ForegroundColor $colors['info']
Write-Host ""
