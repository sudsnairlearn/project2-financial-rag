<#
.SYNOPSIS
Creates a local Python virtual environment and installs project dependencies.

Usage:
  .\setup_env.ps1
#>

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $scriptDir ".venv"
$requirementsPath = Join-Path $scriptDir "requirements.txt"

if (-Not (Test-Path $requirementsPath)) {
    throw "requirements.txt not found in $scriptDir"
}

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    throw "Python executable not found in $venvPath. Ensure python is on PATH."
}

Write-Host "Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
& $pythonExe -m pip install -r $requirementsPath

Write-Host "\nSetup complete. Activate the environment with:"
Write-Host "  .\ .venv\Scripts\Activate.ps1"
