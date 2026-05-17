param(
    [string[]]$Files = @(
        'F:\Sparse_GS\overnight_logs\12_blender_mic_n8_w3_aggrprune_long_v7_widerpseudo_x60k.log',
        'F:\Sparse_GS\overnight_logs\13_blender_lego_n8_w3_aggrprune_long_v7_widerpseudo_x60k.log'
    )
)

foreach ($f in $Files) {
    if (-not (Test-Path $f)) { continue }
    Write-Host ""
    Write-Host ("===== " + (Split-Path $f -Leaf) + " =====") -ForegroundColor Cyan
    # Get-Content 自动识别 UTF-16/UTF-8/BOM
    Get-Content $f | Where-Object {
        $_ -match '^\[eval @' -or
        $_ -match '^\[test\] test/psnr' -or
        $_ -match '^\[pre-prune-test'
    }
}
