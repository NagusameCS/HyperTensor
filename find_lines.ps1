$f = "C:\Users\legom\TensorOS\HyperTensor\host\main.c"
$lines = Get-Content $f

# Find the target lines
$startIdx = -1
$endIdx = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "Tokenize with BOS prepended") {
        $startIdx = $i - 1  # include the blank line before comment
        # Find the closing brace (if (n_ctx <= 0...
        for ($j = $i; $j -lt [Math]::Min($i + 25, $lines.Count); $j++) {
            if ($lines[$j] -match "free\(ctx\); free\(drafts\); return -1;" -and $j -gt $i + 10) {
                $endIdx = $j
                break
            }
        }
        break
    }
}

Write-Host "Start: $startIdx, End: $endIdx"
if ($startIdx -ge 0 -and $endIdx -ge 0) {
    Write-Host "Lines to replace:"
    $lines[$startIdx..$endIdx] | ForEach-Object { Write-Host "  $_" }
}
