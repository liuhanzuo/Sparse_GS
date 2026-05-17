$rows = @()
Get-ChildItem F:\Sparse_GS\outputs -Directory | ForEach-Object {
    $m = Join-Path $_.FullName 'metrics.json'
    if (Test-Path $m) {
        try {
            $j = Get-Content $m -Raw | ConvertFrom-Json
            $rows += [PSCustomObject]@{
                Name     = $_.Name
                PSNR     = [math]::Round($j.metrics.'test/psnr', 3)
                SSIM     = [math]::Round($j.metrics.'test/ssim', 4)
                LPIPS    = [math]::Round($j.metrics.'test/lpips', 4)
                N        = $j.num_gaussians
                Wall_min = [math]::Round($j.wall_clock_sec / 60, 1)
            }
        } catch {}
    }
}
Write-Host ""
Write-Host "================ COMPLETED EXPERIMENTS ================" -ForegroundColor Cyan
$rows | Sort-Object Name | Format-Table -AutoSize

Write-Host ""
Write-Host "================ RUNNING PROCESSES ====================" -ForegroundColor Cyan
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) { $py | Format-Table Id, StartTime, ProcessName -AutoSize } else { Write-Host "(no python running)" }

Write-Host ""
Write-Host "================ RECENT LOG TAIL ======================" -ForegroundColor Cyan
$logs = Get-ChildItem F:\Sparse_GS\overnight_logs\*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 3
foreach ($lf in $logs) {
    Write-Host ""
    Write-Host ("--- " + $lf.Name + "  (" + $lf.LastWriteTime + ", " + [math]::Round($lf.Length/1KB,1) + "KB) ---") -ForegroundColor Yellow
    Get-Content $lf.FullName -Tail 6
}
$rows | Sort-Object Name | Format-Table -AutoSize
