# Launch the LLFF sweep detached so the controlling terminal is free.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_llff_sweep.ps1

$ErrorActionPreference = 'Stop'
$root = 'd:\SSL\sparse_gs'
$python = 'C:\Python314\python.exe'
$script = Join-Path $root 'scripts\run_llff_sweep.py'
$outLog = Join-Path $root 'outputs\_llff_sweep_master.log'
$errLog = Join-Path $root 'outputs\_llff_sweep_master.err'

# Only run the still-missing 4 scenes. The sweep script itself will skip
# already-done runs via metrics.json check, so this is belt-and-braces.
$scenes = @('leaves', 'orchids', 'room', 'trex')

$arglist = @($script, '--scenes') + $scenes
$proc = Start-Process -FilePath $python -ArgumentList $arglist `
    -WorkingDirectory $root `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden -PassThru

Write-Host ("launched PID={0}" -f $proc.Id)
Write-Host ("master log: {0}" -f $outLog)
Write-Host ("err log:    {0}" -f $errLog)
