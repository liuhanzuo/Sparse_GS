#requires -Version 5.1
# ============================================================
# run_ship_ablation.ps1 — Ship 单变量 ablation
# ============================================================
# 跑 4 个 run，每个 ~1 小时（45k iter），共约 4 小时。
# 全部以 ship v7 x45k 为基线，每次只动一个变量，最后一个跑 v8_hard 全套。
#
#   1. baseline                    : v7 x45k (复跑一次确保 seed/环境一致)
#   2. A1 + random_bg              : v7 + random_background_aug only
#   3. A2 + conservative_prune     : v7 + 收紧 densify only
#   4. A3 + weak_pvd               : v7 + 弱 PVD only
#   5. A4 v8_hard_full             : v8_hard 全套（rand_bg + 弱PVD + 收紧 + sh=2 + depth×0.5）
#
# 运行命令：
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\Sparse_GS\run_ship_ablation.ps1
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("ship_ablation_$Stamp.log")
"== ship_ablation start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master
"   estimated total: ~5 hours (5 runs x ~1h)" | Tee-Object -FilePath $Master -Append

$Configs = @(
  'configs\_w3_aggrprune\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A1_rand_bg.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A2_conservative_prune.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\ablation_ship_A3_weak_pvd.yaml',
  'configs\_w3_aggrprune\v8_hard\blender_ship_n8_v8_hard_x45k.yaml'
)

$Results = @()
$Total   = $Configs.Count
$Idx     = 0
$AllStart = Get-Date

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  $log  = Join-Path $LogDir ("ablation_{0:00}_{1}.log" -f $Idx, $name)

  $start = Get-Date
  $hdr   = "[$Idx/$Total] $(Get-Date -Format 'HH:mm:ss') START $name"
  $hdr | Tee-Object -FilePath $Master -Append
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Write-Host $hdr -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
    # tqdm 进度条直通 console；stdout 用 Tee 双写到日志文件
    & $Python -u "$Repo\scripts\train.py" --config "$Repo\$cfg" | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
  } catch {
    $code = -1
    $_ | Out-File -FilePath $log -Append -Encoding utf8
  }

  $elapsed = (Get-Date) - $start
  $mins    = [math]::Round($elapsed.TotalMinutes, 1)

  $outDir = Join-Path $Repo ("outputs\$name")
  $metric = Join-Path $outDir 'metrics.json'
  if (Test-Path $metric) {
    try {
      $j = Get-Content $metric -Raw | ConvertFrom-Json
      $psnr  = $j.metrics.'test/psnr'
      $ssim  = $j.metrics.'test/ssim'
      $lpips = $j.metrics.'test/lpips'
      $line = ("[{0}/{1}] DONE in {2} min  PSNR={3:F4}  SSIM={4:F4}  LPIPS={5:F4}  ({6})" -f
               $Idx, $Total, $mins, $psnr, $ssim, $lpips, $name)
    } catch {
      $line = "[$Idx/$Total] DONE in $mins min  (metrics.json unparsable) ($name)"
    }
  } else {
    $line = "[$Idx/$Total] FAILED in $mins min  exit=$code  no metrics.json  ($name)  see $log"
  }

  $line | Tee-Object -FilePath $Master -Append
  Write-Host ""
  Write-Host $line -ForegroundColor Yellow
  $Results += $line
}

$totalMin = [math]::Round(((Get-Date) - $AllStart).TotalMinutes, 1)
"" | Tee-Object -FilePath $Master -Append
"== ship_ablation done in $totalMin min: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All ship-ablation runs done. Master log: $Master" -ForegroundColor Green
