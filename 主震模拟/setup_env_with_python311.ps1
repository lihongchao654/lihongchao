# PowerShell Script to setup Python environment and run seismic simulation
# This script will:
# 1. Remove old virtual environment (optional, commented by default)
# 2. Create a new .venv with Python 3.11
# 3. Activate the environment
# 4. Upgrade pip, setuptools, wheel
# 5. Install required packages: numpy, pandas, scipy, matplotlib
# 6. Run the seismic simulation script
# 7. Save output to run_output.txt

# Set script execution policy for this session (allows running scripts)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Define paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $scriptDir ".venv"
$pythonScript = Join-Path $scriptDir "人工模拟地震动.py"
$outputFile = Join-Path $scriptDir "run_output.txt"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Environment Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Python 3.11 is available
Write-Host "[Step 1] Checking for Python 3.11..." -ForegroundColor Yellow
$pythonCmd = $null
try {
    $pythonOutput = python3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "python3.11"
        Write-Host "✓ Found Python 3.11: $pythonOutput" -ForegroundColor Green
    }
}
catch {
    Write-Host "✗ Python 3.11 not found, trying 'python'" -ForegroundColor Yellow
    try {
        $pythonOutput = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "python"
            Write-Host "✓ Found Python: $pythonOutput" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "✗ Python not found in PATH. Please install Python 3.11 or higher." -ForegroundColor Red
        exit 1
    }
}

# Step 2: Remove old venv (optional - comment out if you want to keep it)
# Write-Host ""
# Write-Host "[Step 2] Removing old virtual environment..." -ForegroundColor Yellow
# if (Test-Path $venvPath) {
#     Remove-Item -Recurse -Force $venvPath
#     Write-Host "✓ Old venv removed" -ForegroundColor Green
# } else {
#     Write-Host "✓ No old venv found" -ForegroundColor Green
# }

Write-Host ""
Write-Host "[Step 2] Creating new virtual environment..." -ForegroundColor Yellow
if (Test-Path $venvPath) {
    Write-Host "✓ Virtual environment already exists at $venvPath" -ForegroundColor Green
} else {
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "[Step 3] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
& $activateScript
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Step 4: Upgrade pip, setuptools, wheel
Write-Host ""
Write-Host "[Step 4] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip, setuptools, wheel upgraded successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to upgrade pip/setuptools/wheel" -ForegroundColor Red
}

# Step 5: Install required packages
Write-Host ""
Write-Host "[Step 5] Installing required packages..." -ForegroundColor Yellow
Write-Host "  Installing: numpy, pandas, scipy, matplotlib"
pip install numpy pandas scipy matplotlib -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install packages" -ForegroundColor Red
}

# Step 6: Run the Python script and capture output
Write-Host ""
Write-Host "[Step 6] Running seismic simulation script..." -ForegroundColor Yellow
Write-Host "  Script: $pythonScript"
Write-Host "  Output will be saved to: $outputFile"
Write-Host ""
Write-Host "--- Script Output ---" -ForegroundColor Cyan

# Run script and save output
$outputContent = ""
try {
    $output = & python $pythonScript 2>&1
    $outputContent = $output | Out-String
    $LASTEXITCODE_Local = $LASTEXITCODE
}
catch {
    $outputContent = $_.Exception.Message
    $LASTEXITCODE_Local = 1
}

# Display output in console
Write-Host $outputContent -ForegroundColor White

# Save output to file
$outputContent | Out-File -FilePath $outputFile -Encoding UTF8
Write-Host ""
Write-Host "--- End of Script Output ---" -ForegroundColor Cyan
Write-Host ""

# Step 7: Summary
Write-Host "[Step 7] Summary" -ForegroundColor Yellow
if ($LASTEXITCODE_Local -eq 0) {
    Write-Host "✓ Script executed successfully!" -ForegroundColor Green
    Write-Host "✓ Output saved to: $outputFile" -ForegroundColor Green
} else {
    Write-Host "✗ Script execution had errors (exit code: $LASTEXITCODE_Local)" -ForegroundColor Yellow
    Write-Host "✓ Error output saved to: $outputFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Deactivate venv (optional, can be commented out)
# deactivate
