#requires -Version 5.1
# ============================================================
# run_remaining.ps1 — 5 个简单场景 + drums 最后
# ============================================================
# 跑顺序（drums 上次跑飞了高斯爆炸到 4.6M, 放最后；前 5 个先稳住战果）：
#   1. ficus      v7_widerpseudo_x60k    (细长结构, 但小规模, 应该最快)
#   2. mic        v7_widerpseudo_x60k    (薄结构)
#   3. lego       v7_widerpseudo_x60k    (中等复杂度)
#   4. materials  v7_widerpseudo_x60k    (反射材质)
#   5. ship       v7_widerpseudo_x60k    (复杂)
#   6. drums      v7_widerpseudo_x60k    (最难, 上次失控)
#
# 关键改动 vs run_overnight.ps1：
#   - tqdm 进度条直通 console（让你能在 terminal 实时看到 step 进度）
#   - 同时也写日志（用 Tee-Object 双写）
#   - 已有结果的 chair 不重跑
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("remaining_$Stamp.log")
"== remaining start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master

$Configs = @(
  'configs\_w3_aggrprune\blender_ficus_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_mic_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_lego_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
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
  # 用 N+10 起编号, 避开和 overnight_logs 里 01..07 撞名
  $log  = Join-Path $LogDir ("{0:00}_{1}.log" -f ($Idx + 10), $name)

  $start = Get-Date
  $hdr   = "[$Idx/$Total] $(Get-Date -Format 'HH:mm:ss') START $name"
  $hdr | Tee-Object -FilePath $Master -Append
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Write-Host $hdr -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
    # 关键: tqdm 走 stderr, 不重定向 -> 直接刷到 console (动态进度条 \r 生效)
    #       stdout 是训练 print 日志, 用 Tee-Object 双写 (console + 日志文件)
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
      $psnr  = if ($j.test) { $j.test.psnr  } else { $null }
      $ssim  = if ($j.test) { $j.test.ssim  } else { $null }
      $lpips = if ($j.test) { $j.test.lpips } else { $null }
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
"== remaining end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All done. Master log: $Master" -ForegroundColor Green
