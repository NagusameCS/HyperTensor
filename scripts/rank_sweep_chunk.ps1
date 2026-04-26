param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('baseline', '1024', '1536', '2048')]
    [string]$Chunk,
    [string]$Model = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe = "C:\Users\legom\HyperTensor\build_host\geodessical.exe",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $OutDir) {
    $OutDir = Join-Path ".\\benchmarks" ("whitepaper_finalize_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$prompts = @(
    @{ name = 'coding'; text = 'Write a Python function that returns prime numbers up to n.' },
    @{ name = 'reasoning'; text = 'Explain why gradient clipping helps stabilize training in deep networks.' },
    @{ name = 'factual'; text = 'Summarize how TCP congestion control works in modern networks.' },
    @{ name = 'creative'; text = 'Write a short sci-fi paragraph about a city powered by ocean tides.' }
)
$tokensList = @(128, 256)

foreach ($prompt in $prompts) {
    foreach ($tokens in $tokensList) {
        if ($Chunk -eq 'baseline') {
            $label = "baseline_{0}_{1}" -f $prompt.name, $tokens
            $args = @('--temp', '0', '-p', $prompt.text, '-n', "$tokens")
        } else {
            $label = "grc_k{0}_{1}_{2}" -f $Chunk, $prompt.name, $tokens
            $args = @('--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca', '--axex-compress-rank', $Chunk, '--temp', '0', '-p', $prompt.text, '-n', "$tokens")
        }

        $stdoutPath = Join-Path $OutDir ($label + '.txt')
        $stderrPath = Join-Path $OutDir ($label + '_err.txt')
        if (Test-Path $stdoutPath) { continue }

        & $Exe $Model @args 1> $stdoutPath 2> $stderrPath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed chunk case: $label"
        }
    }
}

Write-Host "CHUNK_DONE=$Chunk"
Write-Host "OUTDIR=$((Resolve-Path $OutDir).Path)"
