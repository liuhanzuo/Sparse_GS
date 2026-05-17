#requires -Version 5.1
# ============================================================
# run_v8_hard.ps1 — 一键串行跑 ship → materials → drums 的 v8_hard
# ============================================================
# 三个"难场景"的 v8_hard x45k 配方串行执行，预计每个 ~1 小时，共 ~3 小时。
#
# 顺序选择 ship → materials → drums 的原因：
#   - ship    用 v8_hard 改动最对症（反射 + densify 爆炸最严重），先验证。
#   - materials 与 ship 共享 sh_degree=2 + depth ×0.3，结果可类比。
#   - drums    保持 sh=3 + depth=default，是相对保守的对照。
#
# **强烈建议**先跑：
#     powershell -NoProfile -ExecutionPolicy Bypass -File F:\Sparse_GS\run_smoke_random_bg.ps1
# 通过 smoke 的 |Δ| ≤ 1 dB 判定后再启动本脚本。
#
# 运行命令：
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\Sparse_GS\run_v8_hard.ps1
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("v8_hard_$Stamp.log")
"== v8_hard start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master
"   estimated total: ~3 hours (3 hard scenes x ~1h)" | Tee-Object -FilePath $Master -Append

$Configs = @(
  'configs\_w3_aggrprune\v8_hard\blender_ship_n8_v8_hard_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\blender_materials_n8_v8_hard_x45k.yaml',
  'configs\_w3_aggrprune\v8_hard\blender_drums_n8_v8_hard_x45k.yaml'
)

# v7 baseline 数据（来自此前训练的 metrics.json，用于直接对比 ΔPSNR）
$BaselineV7 = @{
  'blender_ship_n8_v8_hard_x45k'      = @{ scene='ship';      v7_psnr=16.99; v7_ssim=0.7267; v7_lpips=0.2929 }
  'blender_materials_n8_v8_hard_x45k' = @{ scene='materials'; v7_psnr=$null; v7_ssim=$null;  v7_lpips=$null  }
  'blender_drums_n8_v8_hard_x45k'     = @{ scene='drums';     v7_psnr=$null; v7_ssim=$null;  v7_lpips=$null  }
}

$Results = @()
$Total = $Configs.Count
$Idx = 0
$AllStart = Get-Date

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  $log  = Join-Path $LogDir ("v8_hard_{0:00}_{1}.log" -f $Idx, $name)

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
      $psnr  = [double]$j.metrics.'test/psnr'
      $ssim  = [double]$j.metrics.'test/ssim'
      $lpips = [double]$j.metrics.'test/lpips'

      $base = $BaselineV7[$name]
      $deltaStr = ''
      if ($null -ne $base.v7_psnr) {
        $dp = $psnr - $base.v7_psnr
        $deltaStr = ("  vs v7 ΔPSNR={0:+0.00;-0.00;+0.00}" -f $dp)
      }
      $line = ("[{0}/{1}] DONE in {2} min  PSNR={3:F4}  SSIM={4:F4}  LPIPS={5:F4}{6}  ({7})" -f
               $Idx, $Total, $mins, $psnr, $ssim, $lpips, $deltaStr, $name)
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
"== v8_hard done in $totalMin min: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master -Append
"== summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

Write-Host ""
Write-Host "All v8_hard runs done. Master log: $Master" -ForegroundColor Green
