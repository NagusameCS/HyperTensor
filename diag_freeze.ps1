$ErrorActionPreference = 'SilentlyContinue'
Write-Host "=== Error/Critical events last 24h, grouped ==="
Get-WinEvent -LogName System -MaxEvents 1000 |
    Where-Object { $_.LevelDisplayName -in 'Error','Critical' -and $_.TimeCreated -gt (Get-Date).AddHours(-24) } |
    Group-Object ProviderName, Id |
    Sort-Object Count -Descending |
    Select-Object Count, Name |
    Format-Table -AutoSize | Out-String -Width 200

Write-Host "=== Display/GPU/Kernel-Power/BugCheck specific ==="
Get-WinEvent -LogName System -MaxEvents 2000 |
    Where-Object {
        $_.TimeCreated -gt (Get-Date).AddHours(-24) -and
        ($_.ProviderName -match 'nvlddmkm|Display|Kernel-Power|BugCheck|WHEA|nvidia|Kernel-PnP') -and
        $_.LevelDisplayName -in 'Error','Critical','Warning'
    } |
    Select-Object TimeCreated, LevelDisplayName, ProviderName, Id |
    Format-Table -AutoSize | Out-String -Width 200

Write-Host "=== Top 10 most recent errors (truncated msg) ==="
Get-WinEvent -LogName System -MaxEvents 1000 |
    Where-Object { $_.LevelDisplayName -in 'Error','Critical' -and $_.TimeCreated -gt (Get-Date).AddHours(-24) } |
    Sort-Object TimeCreated -Descending |
    Select-Object -First 15 |
    ForEach-Object {
        $msg = ($_.Message -replace "`r?`n", ' ').Substring(0, [Math]::Min(160, $_.Message.Length))
        "{0}  {1,-8} {2,-35} id={3}  {4}" -f $_.TimeCreated, $_.LevelDisplayName, $_.ProviderName, $_.Id, $msg
    }
