#requires -Version 5.1
# ============================================================
# run_smoke_random_bg.ps1 — 1-min smoke for v8_hard random_background
# ============================================================
# 在 hotdog 上 ON / OFF 各跑 1000 iter，确认实现正确性。
# 预期：两次 1000-iter 的 val/psnr 差距 ≤ 1 dB（rand_bg 不应让训练崩）。
#
# 总耗时约 2-4 分钟，必须在跑大任务（run_v8_hard.ps1）之前先验证。
#
# 运行命令：
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\Sparse_GS\run_smoke_random_bg.ps1
# ============================================================

$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\pc\miniconda3\envs\gaussian_splatting\python.exe'
$Repo   = 'F:\Sparse_GS'
$LogDir = Join-Path $Repo 'overnight_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Master = Join-Path $LogDir ("smoke_random_bg_$Stamp.log")
"== smoke_random_bg start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" | Tee-Object -FilePath $Master

$Configs = @(
  'configs\_w3_aggrprune\v8_hard\ablation\smoke_hotdog_rand_bg_OFF.yaml',
  'configs\_w3_aggrprune\v8_hard\ablation\smoke_hotdog_rand_bg_ON.yaml'
)

$Results = @()
$Total = $Configs.Count
$Idx = 0

foreach ($cfg in $Configs) {
  $Idx++
  $name = [IO.Path]::GetFileNameWithoutExtension($cfg)
  $log  = Join-Path $LogDir ("smoke_{0:00}_{1}.log" -f $Idx, $name)

  $start = Get-Date
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor Cyan
  Write-Host "[$Idx/$Total] START $name" -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor Cyan

  try {
    & $Python -u "$Repo\scripts\train.py" --config "$Repo\$cfg" | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
  } catch {
    $code = -1
    $_ | Out-File -FilePath $log -Append -Encoding utf8
  }

  $elapsed = (Get-Date) - $start
  $mins = [math]::Round($elapsed.TotalMinutes, 2)

  $outDir = Join-Path $Repo ("outputs\$name")
  $metric = Join-Path $outDir 'metrics.json'
  if (Test-Path $metric) {
    try {
      $j = Get-Content $metric -Raw | ConvertFrom-Json
      $psnr = $j.metrics.'test/psnr'
      $line = ("[{0}/{1}] DONE in {2} min  test/PSNR={3:F4}  ({4})" -f $Idx, $Total, $mins, $psnr, $name)
    } catch {
      $line = "[$Idx/$Total] DONE in $mins min  (metrics.json unparsable)  ($name)"
    }
  } else {
    $line = "[$Idx/$Total] FAILED in $mins min  exit=$code  no metrics.json  ($name)"
  }
  $line | Tee-Object -FilePath $Master -Append
  Write-Host $line -ForegroundColor Yellow
  $Results += $line
}

# 解析最后一次 val/psnr@1000（从 OFF / ON 的 eval_log.jsonl 读取最后一条）
function Get-FinalValPSNR($name) {
  $jl = Join-Path $Repo ("outputs\$name\eval_log.jsonl")
  if (-not (Test-Path $jl)) { return $null }
  $last = Get-Content $jl | Where-Object { $_ -match '"val/psnr"' } | Select-Object -Last 1
  if (-not $last) { return $null }
  try {
    $obj = $last | ConvertFrom-Json
    return [double]$obj.'val/psnr'
  } catch { return $null }
}

$psnrOff = Get-FinalValPSNR 'ablation_hotdog_smoke_rand_bg_OFF'
$psnrOn  = Get-FinalValPSNR 'ablation_hotdog_smoke_rand_bg_ON'

"" | Tee-Object -FilePath $Master -Append
"== smoke summary ==" | Tee-Object -FilePath $Master -Append
$Results | ForEach-Object { $_ | Tee-Object -FilePath $Master -Append }

if ($null -ne $psnrOff -and $null -ne $psnrOn) {
  $delta = $psnrOn - $psnrOff
  $deltaStr = ("OFF val/psnr={0:F3}  ON val/psnr={1:F3}  delta={2:F3} dB" -f $psnrOff, $psnrOn, $delta)
  $deltaStr | Tee-Object -FilePath $Master -Append
  if ([math]::Abs($delta) -le 1.0) {
    "[OK] random_bg implementation looks healthy (|delta| <= 1 dB)" | Tee-Object -FilePath $Master -Append
    Write-Host "[OK] smoke passed." -ForegroundColor Green
  } else {
    "[WARN] |delta| > 1 dB — inspect cam.alpha / GT recomposition before launching big runs!" | Tee-Object -FilePath $Master -Append
    Write-Host "[WARN] smoke delta too large." -ForegroundColor Red
  }
} else {
  "[WARN] could not read val/psnr from eval_log.jsonl" | Tee-Object -FilePath $Master -Append
}

Write-Host ""
Write-Host "Smoke done. Master log: $Master" -ForegroundColor Green
