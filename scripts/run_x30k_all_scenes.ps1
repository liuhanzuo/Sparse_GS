#requires -Version 5.1
<#
.SYNOPSIS
  按顺序跑 7 个剩余场景的 v6_pgan70_x30k 训练（hotdog 已完成，跳过）。

.DESCRIPTION
  - 每个场景在当前可见的命令行中前台运行（不开 Start-Process / 后台进程）。
  - 单场景训练日志单独写到 outputs/logs/x30k_<scene>.log。
  - 任意场景失败时，仍继续尝试后续场景，最后汇总状态。
  - 训练完成后自动调用 SOTA 收集脚本刷新 outputs/_sota/。

.USAGE
  在 PowerShell 中：
    .\run_x30k_all_scenes.ps1                  # 跑全部 7 个剩余场景
    .\run_x30k_all_scenes.ps1 -Scenes drums    # 只跑 drums
    .\run_x30k_all_scenes.ps1 -SkipHotdog:$false  # 把 hotdog 也跑（默认跳过，因为已完成）
#>
param(
    [string[]]$Scenes = @('drums','chair','lego','ficus','materials','mic','ship'),
    [switch]$SkipHotdog = $true,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$logDir = Join-Path $projectRoot 'outputs\logs'
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# 确保 PYTHONIOENCODING 设置（中文/特殊字符日志）
$env:PYTHONIOENCODING = 'utf-8'

# hotdog 已经做过 x30k，默认跳过；如果显式传 -SkipHotdog:$false 则强制跑一次
if (-not $SkipHotdog) {
    $Scenes = @('hotdog') + $Scenes
}

$results = @()
$startWall = Get-Date

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  x30k batch run on $($Scenes.Count) scenes" -ForegroundColor Cyan
Write-Host "  scenes : $($Scenes -join ', ')" -ForegroundColor Cyan
Write-Host "  log dir: $logDir" -ForegroundColor Cyan
Write-Host "  dry-run: $DryRun" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

foreach ($scene in $Scenes) {
    $cfg = "configs/_w3_aggrprune/blender_${scene}_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml"
    $absCfg = Join-Path $projectRoot $cfg
    if (-not (Test-Path $absCfg)) {
        Write-Host "[SKIP] $scene : config not found ($cfg)" -ForegroundColor Yellow
        $results += [pscustomobject]@{ scene = $scene; status = 'missing_config'; sec = 0 }
        continue
    }

    $logFile = Join-Path $logDir "x30k_${scene}.log"
    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Green
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] >>> RUN scene=$scene" -ForegroundColor Green
    Write-Host "    config = $cfg"
    Write-Host "    log    = $logFile"
    Write-Host "------------------------------------------------------------" -ForegroundColor Green

    if ($DryRun) {
        Write-Host "[DRY-RUN] would run: python -u scripts/train.py --config $cfg" -ForegroundColor Magenta
        $results += [pscustomobject]@{ scene = $scene; status = 'dry_run'; sec = 0 }
        continue
    }

    $sceneStart = Get-Date

    # 在当前 console 中前台运行；同步把 stdout/stderr 重定向到日志文件，
    # 同时仍然在屏幕上看得到（Tee-Object）。
    & python -u scripts/train.py --config $cfg 2>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE

    $sceneSec = [int]((Get-Date) - $sceneStart).TotalSeconds
    if ($exitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] <<< OK  scene=$scene  ($sceneSec s)" -ForegroundColor Green
        $results += [pscustomobject]@{ scene = $scene; status = 'ok'; sec = $sceneSec }
    } else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] <<< FAIL scene=$scene  exit=$exitCode  ($sceneSec s)" -ForegroundColor Red
        $results += [pscustomobject]@{ scene = $scene; status = "fail($exitCode)"; sec = $sceneSec }
    }

    # 每跑完一个场景就刷新一次 SOTA 表格（即使失败也尝试，方便看进度）
    try {
        & python -u scripts/collect_x30k_sota.py
    } catch {
        Write-Host "[warn] SOTA collector failed: $_" -ForegroundColor Yellow
    }
}

$totalSec = [int]((Get-Date) - $startWall).TotalSeconds

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  BATCH SUMMARY  (total $totalSec s = $([int]($totalSec/60)) min)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$results | Format-Table -AutoSize

# 最终再刷新一次 SOTA
if (-not $DryRun) {
    Write-Host ""
    Write-Host "[final] refreshing SOTA table..." -ForegroundColor Cyan
    & python -u scripts/collect_x30k_sota.py
}
