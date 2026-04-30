# queue_qwen_after_gemma.ps1
# Waits for gemma4:31b pull to finish, then starts qwen3.5:35b pull.
$logDir = 'C:\Users\legom\HyperTensor\benchmarks'
$ollamaExe = 'C:\Users\legom\AppData\Local\Programs\Ollama\ollama.exe'

Write-Host "[queue] Waiting for gemma4:31b to finish..."
while ($true) {
    $proc = Get-Process | Where-Object Name -eq 'ollama' | Where-Object Id -ne (Get-Process | Where-Object Name -eq 'ollama' | Sort-Object Id | Select-Object -First 1).Id | Select-Object -First 1
    # Check if gemma pull process (PID in pids file) is still alive
    $pidLine = Get-Content "$logDir\ollama_pull_pids.txt" -ErrorAction SilentlyContinue | Select-String 'gemma_pid=(\d+)'
    if ($pidLine) {
        $gemmaPid = [int]$pidLine.Matches[0].Groups[1].Value
        if (-not (Get-Process -Id $gemmaPid -ErrorAction SilentlyContinue)) {
            Write-Host "[queue] gemma pull process exited. Checking if complete..."
            break
        }
    }
    # Also check via ollama list
    $models = & $ollamaExe list 2>$null
    if ($models -match 'gemma4:31b') {
        Write-Host "[queue] gemma4:31b confirmed in ollama list."
        break
    }
    Start-Sleep -Seconds 30
}

Write-Host "[queue] Starting qwen3.5:35b pull..."
$p2 = Start-Process -FilePath $ollamaExe `
    -ArgumentList 'pull','qwen3.5:35b' `
    -RedirectStandardOutput "$logDir\ollama_pull_qwen35_35b.log" `
    -RedirectStandardError  "$logDir\ollama_pull_qwen35_35b.err" `
    -PassThru -WindowStyle Hidden
"qwen_pid=$($p2.Id)" | Tee-Object -Append "$logDir\ollama_pull_pids.txt"
Write-Host "[queue] qwen3.5:35b pull started, pid=$($p2.Id)"
