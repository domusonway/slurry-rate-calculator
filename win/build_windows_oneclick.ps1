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
    $candidates = @(
        @("python"),
        @("python3"),
        @("py", "-3")
    )

    foreach ($cand in $candidates) {
        $exe = $cand[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
            continue
        }

        try {
            $args = @()
            if ($cand.Count -gt 1) {
                $args += $cand[1..($cand.Count - 1)]
            }
            $args += "--version"
            & $exe @args | Out-Null
            return $cand
        } catch {
            Write-Host "候选 Python 命令不可用：$($cand -join ' ')，尝试下一个"
        }
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
    try {
        $pythonVersionArgs = @()
        if ($pythonCmd.Count -gt 1) {
            $pythonVersionArgs = $pythonCmd[1..($pythonCmd.Count - 1)]
        }
        $pythonVersionArgs += "--version"
        $verText = (& $pythonCmd[0] @pythonVersionArgs 2>&1 | Out-String).Trim()
        Write-Output "检测到 Python: $verText"
        $match = [regex]::Match($verText, "\\d+\\.\\d+")
        if (-not $match.Success) {
            throw "无法解析 Python 版本：$verText"
        }
        if ([version]$match.Value -lt [version]"3.10") {
            throw "当前 Python 版本为 $verText，需 >=3.10。"
        }
    } catch {
        throw "Python 版本检查失败：$($pythonCmd -join ' ')"
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
    $pyInstallerArgs = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name", "SlurryRateCalculator",
        "--windowed",
        "--collect-all", "streamlit",
        "--add-data", ((Join-Path $root "img") + ";img"),
        "--hidden-import", "cv2",
        "app.py"
    )

    $imgDir = Join-Path $root "img"
    if (-not (Test-Path $imgDir)) {
        throw "打包缺失资源目录：$imgDir"
    }

    $streamlitDir = Join-Path $root ".streamlit"
    if (Test-Path $streamlitDir) {
        $pyInstallerArgs += @("--add-data", ($streamlitDir + ";.streamlit"))
    } else {
        Write-Warning ".streamlit 目录不存在，跳过 add-data 处理"
    }

    $manualFile = Join-Path $root "user_manual.md"
    if (Test-Path $manualFile) {
        $pyInstallerArgs += @("--add-data", ($manualFile + ";."))
    } else {
        Write-Warning "user_manual.md 不存在，跳过 add-data 处理"
    }

    Invoke-Python @($pyInstallerArgs)
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
if (Test-Path (Join-Path $root "user_manual.md")) {
    Copy-Item -Path (Join-Path $root "user_manual.md") -Destination $packageDir -Force
}
if (Test-Path (Join-Path $root ".streamlit")) {
    Copy-Item -Path (Join-Path $root ".streamlit") -Destination (Join-Path $packageDir ".streamlit") -Recurse -Force
}
Copy-Item -Path (Join-Path $root "img") -Destination (Join-Path $packageDir "img") -Recurse -Force
Copy-Item -Path (Join-Path $root "win\start.bat") -Destination $packageDir -Force
Copy-Item -Path (Join-Path $root "win\quickstart.bat") -Destination $packageDir -Force

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force

Write-Output "一键部署包已生成：$zipPath"
