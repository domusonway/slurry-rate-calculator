#Requires -Version 5.1
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $root "dist"
$packageDir = Join-Path $distRoot "slurry-rate-calculator-windows-oneclick"
$zipPath = Join-Path $distRoot "slurry-rate-calculator-windows-oneclick.zip"

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        return @("python3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "未检测到 Python。请先安装 Python 3.11+（可直接执行 start.bat 使用 Python 回退版）。"
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    if ($pythonCmd.Count -le 1) {
        & $pythonCmd[0] @Arguments
    } else {
        $args = @($pythonCmd[1..($pythonCmd.Count - 1)])
        $args += $Arguments
        & $pythonCmd[0] @args
    }
}

function Require-PythonVersion {
    if ($pythonCmd.Count -le 1) {
        $verText = & $pythonCmd[0] "-c" "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    } else {
        $verText = & $pythonCmd[0] $pythonCmd[1] "-c" "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    }
    if (-not $verText) {
        throw "无法获取 Python 版本。"
    }
    if ([version]$verText -lt [version]"3.10") {
        throw "当前 Python 版本为 $verText，需 >=3.10。"
    }
}

$pythonCmd = Resolve-PythonCommand
Write-Output "使用 Python: $($pythonCmd -join ' ')"
Require-PythonVersion

New-Item -ItemType Directory -Force -Path $distRoot | Out-Null

if (-not $SkipBuild) {
    Write-Output "安装/升级 PyInstaller..."
    Invoke-Python @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Python @("-m", "pip", "install", "pyinstaller", "streamlit", "opencv-python-headless", "numpy", "streamlit-image-comparison", "streamlit-drawable-canvas")

    Write-Output "开始生成 One-File 可执行文件..."
    Invoke-Python @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name", "SlurryRateCalculator",
        "--windowed",
        "--collect-all", "streamlit",
        "--add-data", ((Join-Path $root "img") + ";img"),
        "--add-data", ((Join-Path $root ".streamlit") + ";.streamlit"),
        "--add-data", ((Join-Path $root "user_manual.md") + ";."),
        "--hidden-import", "cv2",
        "app.py"
    )
}

$exePath = Join-Path $root "dist\SlurryRateCalculator.exe"
if (-not (Test-Path $exePath)) {
    throw "可执行文件未生成：$exePath"
}

if (Test-Path $packageDir) {
    Remove-Item -Recurse -Force $packageDir
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

Copy-Item -Path $exePath -Destination (Join-Path $packageDir "SlurryRateCalculator.exe") -Force
Copy-Item -Path (Join-Path $root "app.py") -Destination $packageDir -Force
Copy-Item -Path (Join-Path $root "requirements.txt") -Destination $packageDir -Force
Copy-Item -Path (Join-Path $root "user_manual.md") -Destination $packageDir -Force
Copy-Item -Path (Join-Path $root ".streamlit") -Destination (Join-Path $packageDir ".streamlit") -Recurse -Force
Copy-Item -Path (Join-Path $root "img") -Destination (Join-Path $packageDir "img") -Recurse -Force
Copy-Item -Path (Join-Path $root "win\start.bat") -Destination $packageDir -Force
Copy-Item -Path (Join-Path $root "win\quickstart.bat") -Destination $packageDir -Force

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force

Write-Output "一键部署包已生成：$zipPath"
