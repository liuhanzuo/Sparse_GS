#requires -Version 5.1
# ============================================================
# run_missing2.ps1 — 只跑 ship + drums
# ============================================================
# 起因：materials 在 v7_widerpseudo 配方下训崩了
#   - val/psnr 从 12.18 一路跌到 11.56（18k步）
#   - train/psnr 30+，gap 巨大 → 严重过拟合 + 颜色跑飞
#   - 高斯数已经 1.67M，超出健康范围
# 决定：跳过 materials，等 ship/drums 跑完后单独给 materials 出保守配置重跑
#
# 跑顺序（ship 难度中等放前，drums 上次失控放最后）：
#   1. ship        v7_widerpseudo_x60k
#   2. drums       v7_widerpseudo_x60k
# 预计 ~3.5h
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("missing2_$Stamp.log")
"== missing2 start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master

$Configs = @(
  'configs\_w3_aggrprune\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_drums_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml'
)

$Results = @()
$Total   = $Configs.Count
$Idx     = 0

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  # 编号 30+ 避免和之前的 01..07 / 11..16 / 21..23 撞
  $log  = Join-Path $LogDir ("{0:00}_{1}.log" -f ($Idx + 30), $name)

  $start = Get-Date
  $hdr   = "[$Idx/$Total] $(Get-Date -Format 'HH:mm:ss') START $name"
  $hdr | Tee-Object -FilePath $Master -Append
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Write-Host $hdr -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
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

"" | Tee-Object -FilePath $Master -Append
"== missing2 end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All done. Master log: $Master" -ForegroundColor Green
