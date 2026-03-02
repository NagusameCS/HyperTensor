# Check sector 1 of the SD card (F:) for TENSOR_OK! pattern
# Sector 1 is at byte offset 512 on the raw disk (PhysicalDrive)
# Since F: starts at partition offset 2048 sectors, sector 1 is BEFORE the partition.
# We need raw disk access OR we check via diskpart.

# Method: Write known data to disk sector 1 from Windows BEFORE Pi test,
# then check if it changed after.

$diskNum = (Get-Partition -DriveLetter F | Get-Disk).Number
Write-Host "SD card is PhysicalDrive$diskNum"
Write-Host "Reading sector 1 (bytes 512-1023) via raw disk..."

# Need admin for raw disk access
try {
    $stream = [System.IO.FileStream]::new("\\.\PhysicalDrive$diskNum", 'Open', 'Read', 'ReadWrite')
    $null = $stream.Seek(512, 'Begin')  # Sector 1
    $buf = New-Object byte[] 512
    $null = $stream.Read($buf, 0, 512)
    $stream.Close()

    Write-Host "`nFirst 64 bytes (hex):"
    ($buf[0..63] | ForEach-Object { '{0:X2}' -f $_ }) -join ' '

    Write-Host "`nFirst 64 bytes (ASCII):"
    [System.Text.Encoding]::ASCII.GetString($buf, 0, 64)

    $str = [System.Text.Encoding]::ASCII.GetString($buf, 0, 10)
    if ($str -eq "TENSOR_OK!") {
        Write-Host "`n*** SUCCESS! TENSOR_OK! found at sector 1! ***" -ForegroundColor Green
    } else {
        Write-Host "`n*** Pattern NOT found at sector 1 ***" -ForegroundColor Red
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Try running as Administrator" -ForegroundColor Yellow
}
