#requires -Version 5.1
# ============================================================
# run_missing.ps1 — 只跑真正缺失的 3 个场景
# ============================================================
# 当前已完成 (跳过):
#   - hotdog (v6_pgan70 + v7_widerpseudo, 都有)
#   - chair  (v7_widerpseudo)
#   - ficus  (v7_widerpseudo)  PSNR=24.525
#   - mic    (v7_widerpseudo)  PSNR=25.758
#   - lego   (v7_widerpseudo)  PSNR=23.455
#
# 本脚本要跑 (按预估时长升序, drums 放最后):
#   1. materials   v7_widerpseudo_x60k    (反射材质, 中等时长)
#   2. ship        v7_widerpseudo_x60k    (复杂)
#   3. drums       v7_widerpseudo_x60k    (最难, 上次跑飞)
#
# 同样的实时进度条 + Tee 日志策略.
# 预计总时长 ~5h.
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("missing_$Stamp.log")
"== missing start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master

$Configs = @(
  'configs\_w3_aggrprune\blender_materials_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_drums_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml'
)

$Results = @()
$Total   = $Configs.Count
$Idx     = 0

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  # 编号 20+ 避免和之前的 01..07 / 11..16 撞
  $log  = Join-Path $LogDir ("{0:00}_{1}.log" -f ($Idx + 20), $name)

  $start = Get-Date
  $hdr   = "[$Idx/$Total] $(Get-Date -Format 'HH:mm:ss') START $name"
  $hdr | Tee-Object -FilePath $Master -Append
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Write-Host $hdr -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
    # tqdm (stderr) 直通 console 实现动态进度条; stdout (训练日志) 用 Tee 双写
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
"== missing end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All done. Master log: $Master" -ForegroundColor Green
