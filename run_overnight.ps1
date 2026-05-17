#requires -Version 5.1
# ============================================================
# run_overnight.ps1 — 7 连跑 (~9.3 小时)
# ============================================================
# 跑顺序（每个 ~80 min, x60k）：
#   1. chair      v7_widerpseudo_x60k    (验证泛化)
#   2. drums      v7_widerpseudo_x60k    (验证泛化, sparse-view 最难)
#   3. ficus      v7_widerpseudo_x60k    (验证泛化, 细长结构)
#   4. lego       v7_widerpseudo_x60k    (验证泛化)
#   5. materials  v7_widerpseudo_x60k    (验证泛化, 反射材质)
#   6. mic        v7_widerpseudo_x60k    (验证泛化, 薄结构)
#   7. ship       v7_widerpseudo_x60k    (验证泛化)
#
# 选 x60k 而非 x45k 的原因：
#   hotdog v7 在 step=45000 仍未收敛 (最后 1500 步还涨 +0.08 dB),
#   预计 60k 还能再涨 0.2-0.4 dB。所有时间锚点按 ×4/3 等比放大。
#
# 设计原则：
#   - 一个失败不阻塞后续：每个用 try/catch 包裹，并把 stdout/stderr
#     重定向到独立的日志文件，便于明早 grep 查看哪个跑挂了。
#   - 每跑完一个就在 console 打印汇总行，让你回家时一眼能看进度。
#   - 全部跑完后打印最终汇总（含每个的 metrics.json 是否生成）。
# ============================================================

$ErrorActionPreference = 'Continue'    # 一个失败不挂掉整个脚本
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("overnight_$Stamp.log")
"== overnight start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master

$Configs = @(
  'configs\_w3_aggrprune\blender_chair_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_drums_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_ficus_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_lego_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_materials_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_mic_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml',
  'configs\_w3_aggrprune\blender_ship_n8_w3_aggrprune_long_v7_widerpseudo_x60k.yaml'
)

$Results = @()
$Total   = $Configs.Count
$Idx     = 0

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  $log  = Join-Path $LogDir ("{0:00}_{1}.log" -f $Idx, $name)

  $start = Get-Date
  $hdr   = "[$Idx/$Total] $(Get-Date -Format 'HH:mm:ss') START $name"
  $hdr | Tee-Object -FilePath $Master -Append
  Write-Host $hdr -ForegroundColor Cyan

  try {
    & $Python -u "$Repo\scripts\train.py" --config "$Repo\$cfg" *> $log
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
  Write-Host $line -ForegroundColor Yellow
  $Results += $line
}

"" | Tee-Object -FilePath $Master -Append
"== overnight end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All done. Master log: $Master" -ForegroundColor Green
