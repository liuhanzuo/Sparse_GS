#requires -Version 5.1
# ============================================================
# run_overnight_9h.ps1 — 9 hours unattended overnight pipeline
# ============================================================
# 一键串起：smoke → v8_hard 三难场景 → ship+materials ablation → 全量 eval
# → ablation 汇总。预计 ~9.1 h，正好填满 9 h 窗口。
#
# 编排策略（避免重复跑）：
#   STAGE 1 [3 min]  smoke ON/OFF (rand_bg 实现冒烟，挂了则全停)
#   STAGE 2 [~3 h]   run_v8_hard.ps1   ship + materials + drums (v8_hard)
#   STAGE 3 [~4 h]   ship_ablation A0..A3 (v7 baseline + A1 rand_bg + A2 cons_prune + A3 weak_pvd)
#                    *跳过 A4 v8_hard_full 因为 STAGE 2 已经跑了 ship v8_hard
#   STAGE 3.5 [~2 h] materials_ablation A1/A3 (rand_bg + weak_pvd) 平行实验
#                    *回答 "v8_hard 在 materials 上的提升来自哪一项配方"
#   STAGE 4 [<5 min] 把 STAGE 2/3/3.5 的 last.pt 用 eval_ckpt 全量 200 view 复评
#   STAGE 5 [秒级]   ablation_compare.py + collect_metrics.py 输出汇总表
#
# 失败容错：除 STAGE 1 smoke 强校验外，任何后续 run 失败都会写入 master log
# 并继续跑下一项；不会卡住。每个 STAGE 跑完都写时间戳，方便事后定位。
#
# 启动命令（你 9h 出门前贴这一行）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\Sparse_GS\run_overnight_9h.ps1
#
# 推荐：开个 PowerShell 窗口直接跑；锁屏不影响（前几次都验证过）。
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("overnight9h_$Stamp.log")
$AllStart = Get-Date

function Log($msg, $color='White') {
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  $line = "[$ts] $msg"
  $line | Tee-Object -FilePath $Master -Append | Out-Null
  Write-Host $line -ForegroundColor $color
}

Log "== overnight9h start =="
Log "   master log: $Master"

# -----------------------------------------------------------------
# Helper：跑单个 train.py，stdout 双写 console + 单独日志文件
# -----------------------------------------------------------------
function Invoke-TrainRun {
  param(
    [string]$ConfigRel,   # configs\... 相对仓库根
    [string]$Tag,         # 'v8h_01' 之类的前缀，仅用于日志命名
    [int]$Idx,
    [int]$Total
  )
  $name = [IO.Path]::GetFileNameWithoutExtension($ConfigRel)
  $log  = Join-Path $LogDir ("${Tag}_{0:00}_{1}.log" -f $Idx, $name)
  $start = Get-Date

  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Log "[$Tag $Idx/$Total] START $name" 'Cyan'
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
    & $Python -u "$Repo\scripts\train.py" --config "$Repo\$ConfigRel" | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
  } catch {
    $code = -1
    $_ | Out-File -FilePath $log -Append -Encoding utf8
  }
  $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)

  $outDir = Join-Path $Repo ("outputs\$name")
  $metric = Join-Path $outDir 'metrics.json'
  if (Test-Path $metric) {
    try {
      $j = Get-Content $metric -Raw | ConvertFrom-Json
      $psnr  = [double]$j.metrics.'test/psnr'
      $ssim  = [double]$j.metrics.'test/ssim'
      $lpips = [double]$j.metrics.'test/lpips'
      Log ("[$Tag $Idx/$Total] DONE in $mins min  PSNR={0:F4}  SSIM={1:F4}  LPIPS={2:F4}  ($name)" -f $psnr,$ssim,$lpips) 'Yellow'
      return $true
    } catch {
      Log "[$Tag $Idx/$Total] DONE in $mins min  metrics.json unparsable  ($name)" 'Yellow'
      return $true
    }
  } else {
    Log "[$Tag $Idx/$Total] FAILED in $mins min  exit=$code  no metrics.json  ($name)  see $log" 'Red'
    return $false
  }
}

# =================================================================
# STAGE 1  —  Smoke test (~3 min)
# =================================================================
Log "" 
Log ">>> STAGE 1 / 5: smoke (rand_bg ON vs OFF, hotdog 1k iter)" 'Magenta'
$smokeCfgs = @(
  'configs\_w3_aggrprune\v8_hard\ablation\smoke_hotdog_rand_bg_OFF.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\smoke_hotdog_rand_bg_ON.yaml'
)
$si = 0
foreach ($c in $smokeCfgs) { $si++; [void](Invoke-TrainRun -ConfigRel $c -Tag 'smoke' -Idx $si -Total $smokeCfgs.Count) }

# 解析 smoke 的 val/psnr@1000，强校验 |Δ| ≤ 1 dB；不通过则中止后续 STAGE
function Get-FinalValPSNR($name) {
  $jl = Join-Path $Repo ("outputs\$name\eval_log.jsonl")
  if (-not (Test-Path $jl)) { return $null }
  $last = Get-Content $jl | Where-Object { $_ -match '"val/psnr"' } | Select-Object -Last 1
  if (-not $last) { return $null }
  try { return [double]($last | ConvertFrom-Json).'val/psnr' } catch { return $null }
}
$psnrOff = Get-FinalValPSNR 'ablation_hotdog_smoke_rand_bg_OFF'
$psnrOn  = Get-FinalValPSNR 'ablation_hotdog_smoke_rand_bg_ON'

$smokeOK = $false
if ($null -ne $psnrOff -and $null -ne $psnrOn) {
  $delta = $psnrOn - $psnrOff
  Log ("   smoke: OFF={0:F3}  ON={1:F3}  delta={2:+0.000;-0.000;+0.000} dB" -f $psnrOff,$psnrOn,$delta)
  if ([math]::Abs($delta) -le 1.0) {
    Log "   [OK] rand_bg implementation healthy (|delta| <= 1 dB)" 'Green'
    $smokeOK = $true
  } else {
    Log "   [WARN] |delta| > 1 dB, rand_bg implementation may be broken" 'Red'
    Log "   ABORTING all subsequent stages — please inspect smoke log before retrying." 'Red'
  }
} else {
  Log "   [WARN] could not read smoke val/psnr — proceeding anyway (best-effort)" 'Yellow'
  $smokeOK = $true   # 读取失败不一定真挂，让后面继续跑
}

if (-not $smokeOK) {
  Log "== overnight9h aborted at STAGE 1 =="
  exit 1
}

# =================================================================
# STAGE 2  —  v8_hard on 3 hard scenes (~3 h)
# =================================================================
Log ""
Log ">>> STAGE 2 / 5: v8_hard on ship / materials / drums (~3 h)" 'Magenta'
$v8Cfgs = @(
  'configs\_w3_aggrprune\v8_hard\blender_ship_n8_v8_hard_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\blender_materials_n8_v8_hard_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\blender_drums_n8_v8_hard_x45k.yaml'
)
$vi = 0
foreach ($c in $v8Cfgs) { $vi++; [void](Invoke-TrainRun -ConfigRel $c -Tag 'v8h' -Idx $vi -Total $v8Cfgs.Count) }

# =================================================================
# STAGE 3  —  Ship ablation A0/A1/A2/A3 (~3 h, 跳过 A4=v8_hard 防重复)
# =================================================================
Log ""
Log ">>> STAGE 3 / 5: ship single-variable ablation A0..A3 (~3 h)" 'Magenta'
$abCfgs = @(
  'configs\_w3_aggrprune\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A1_rand_bg.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A2_conservative_prune.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A3_weak_pvd.yaml'
)
# 如果 v7 baseline 已经训练过（之前那次 4h23min 跑完的），就跳过 A0 复跑
$v7BaseDir = Join-Path $Repo 'outputs\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x45k'
if (Test-Path (Join-Path $v7BaseDir 'metrics.json')) {
  Log "   v7 ship baseline already trained — skipping A0 to save ~1h" 'Yellow'
  $abCfgs = $abCfgs[1..3]
}
$ai = 0
foreach ($c in $abCfgs) { $ai++; [void](Invoke-TrainRun -ConfigRel $c -Tag 'abl' -Idx $ai -Total $abCfgs.Count) }

# =================================================================
# STAGE 3.5  —  Materials parallel ablation (~2 h, 填满 9h 窗口)
# =================================================================
Log ""
Log ">>> STAGE 3.5 / 5: materials parallel ablation A1/A3 (~2 h)" 'Magenta'
$abMatCfgs = @(
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_materials_A1_rand_bg.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_materials_A3_weak_pvd.yaml'
)
$mi = 0
foreach ($c in $abMatCfgs) { $mi++; [void](Invoke-TrainRun -ConfigRel $c -Tag 'abl_mat' -Idx $mi -Total $abMatCfgs.Count) }

# =================================================================
# STAGE 4  —  Full-coverage eval on every fresh run (last.pt → metrics_full.json)
# =================================================================
Log ""
Log ">>> STAGE 4 / 5: eval_ckpt full 200-view re-evaluation" 'Magenta'

$runsToFullEval = @(
  'blender_ship_n8_v8_hard_x45k',
  'blender_materials_n8_v8_hard_x45k',
  'blender_drums_n8_v8_hard_x45k',
  'ablation_ship_A1_rand_bg',
  'ablation_ship_A2_conservative_prune',
  'ablation_ship_A3_weak_pvd',
  'ablation_materials_A1_rand_bg',
  'ablation_materials_A3_weak_pvd'
)
foreach ($r in $runsToFullEval) {
  $runDir = Join-Path $Repo "outputs\$r"
  $ckpt   = Join-Path $runDir 'ckpts\last.pt'
  if (-not (Test-Path $ckpt)) { Log "   skip $r (no ckpts\last.pt)" 'Yellow'; continue }
  $log = Join-Path $LogDir ("eval_full_$r.log")
  Log "   eval $r ..." 'Cyan'
  try {
    & $Python -u "$Repo\scripts\eval_ckpt.py" --run "$runDir" 2>&1 | Tee-Object -FilePath $log | Out-Null
    Log "   ok" 'Green'
  } catch {
    Log "   eval failed for $r ($($_.Exception.Message))" 'Red'
  }
}

# =================================================================
# STAGE 5  —  Summary tables
# =================================================================
Log ""
Log ">>> STAGE 5 / 5: summary tables" 'Magenta'
try {
  & $Python -u "$Repo\scripts\ablation_compare.py" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir 'ablation_compare.log') | Out-Null
  Log "   ablation_compare.py -> outputs\_ablation_summary.md" 'Green'
} catch { Log "   ablation_compare.py failed: $($_.Exception.Message)" 'Red' }
try {
  & $Python -u "$Repo\scripts\collect_metrics.py" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir 'collect_metrics.log') | Out-Null
  Log "   collect_metrics.py -> outputs\_summary.md / .csv / .json" 'Green'
} catch { Log "   collect_metrics.py failed: $($_.Exception.Message)" 'Red' }

# -----------------------------------------------------------------
$totalMin = [math]::Round(((Get-Date) - $AllStart).TotalMinutes, 1)
Log ""
Log "== overnight9h ALL DONE in $totalMin min =="
Log "   master log : $Master"
Log "   ablation   : $Repo\outputs\_ablation_summary.md"
Log "   summary    : $Repo\outputs\_summary.md"
Write-Host ""
Write-Host "Done. You can come back now :)" -ForegroundColor Green
