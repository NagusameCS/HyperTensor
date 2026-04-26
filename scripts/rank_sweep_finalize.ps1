$ErrorActionPreference = 'Stop'
$MODEL = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
$EXE = ".\build_host\geodessical.exe"
$OUTDIR = ".\benchmarks\whitepaper_rank_complete_20260425_205838"

function Parse-Run([string]$stdoutPath){
  $raw = Get-Content -Raw -Path $stdoutPath
  $mDec = [regex]::Match($raw,'Decode-only:\s*prefill\s*([\d.]+)\s*ms,\s*([\d.]+)\s*tok/s')
  $mGd = [regex]::Match($raw,'\[GD\]\s*(\d+)\s+tokens\s+in\s*([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)')
  $mCompact = [regex]::Match($raw,'\[(\d+)\s+tok,\s*([\d.]+)\s*tok/s,\s*prefill\s*([\d.]+)\s*ms')
  $decode = $null; $prefill = $null; $overall = $null; $genTok = $null
  if($mDec.Success){ $prefill=[double]$mDec.Groups[1].Value; $decode=[double]$mDec.Groups[2].Value }
  if($mGd.Success){ $genTok=[int]$mGd.Groups[1].Value; $overall=[double]$mGd.Groups[3].Value }
  elseif($mCompact.Success){ $genTok=[int]$mCompact.Groups[1].Value; if($null -eq $decode){$decode=[double]$mCompact.Groups[2].Value}; if($null -eq $prefill){$prefill=[double]$mCompact.Groups[3].Value} }
  [pscustomobject]@{decode_tps=$decode; overall_tps=$overall; prefill_ms=$prefill; generated_tokens=$genTok}
}

function Run-Case([string]$label,[string]$prompt,[int]$tokens,[string[]]$extraArgs){
  $safe = ($label -replace '[^a-zA-Z0-9_\-]','_')
  $out = Join-Path $OUTDIR "${safe}.txt"
  $err = Join-Path $OUTDIR "${safe}_err.txt"
  if(Test-Path $out){ return (Parse-Run $out) }
  Write-Host "Running: $label"
  & $EXE $MODEL @extraArgs -p $prompt -n $tokens 1> $out 2> $err
  if($LASTEXITCODE -ne 0){ throw "case failed: $label" }
  Parse-Run $out
}

$prompts = @(
  @{name='coding'; text='Write a Python function that returns prime numbers up to n.'},
  @{name='reasoning'; text='Explain why gradient clipping helps stabilize training in deep networks.'},
  @{name='factual'; text='Summarize how TCP congestion control works in modern networks.'},
  @{name='creative'; text='Write a short sci-fi paragraph about a city powered by ocean tides.'}
)
$tokenList = @(128,256)
$ranks = @(1024,1536,2048)

$rankRows = @()
foreach($p in $prompts){
  foreach($t in $tokenList){
    # Baseline
    $rankRows += [pscustomobject]@{
      label=("baseline_{0}_{1}" -f $p.name,$t)
      prompt=$p.text; tokens=$t; rank=$null
      result=(Run-Case ("baseline_{0}_{1}" -f $p.name,$t) $p.text $t @('--temp','0'))
    }
    # Each rank
    foreach($r in $ranks){
      $result = Run-Case ("grc_k{0}_{1}_{2}" -f $r,$p.name,$t) $p.text $t @('--axex-compress','--axex-attn-only','--axex-skip-o','--axex-weight-pca','--axex-compress-rank',"$r",'--temp','0')
      $rankRows += [pscustomobject]@{
        label=("grc_k{0}_{1}_{2}" -f $r,$p.name,$t)
        prompt=$p.name; tokens=$t; rank=$r
        result=$result
      }
    }
  }
}

# Flatten results
$flatRows = @()
foreach($r in $rankRows){
  $flatRows += [pscustomobject]@{
    label=$r.label; prompt=$r.prompt; tokens=$r.tokens; rank=$r.rank;
    decode_tps=$r.result.decode_tps; overall_tps=$r.result.overall_tps;
    prefill_ms=$r.result.prefill_ms; generated_tokens=$r.result.generated_tokens
  }
}

$rankCsv = Join-Path $OUTDIR 'rank_sweep_raw.csv'
$flatRows | Export-Csv -NoTypeInformation -Path $rankCsv
Write-Host "Rank sweep raw: $rankCsv"

# Relative calculations
$relativeRows = @()
foreach($p in $prompts){
  foreach($t in $tokenList){
    $b = $flatRows | Where-Object { $_.label -eq ("baseline_{0}_{1}" -f $p.name,$t) } | Select-Object -First 1
    foreach($r in $ranks){
      $g = $flatRows | Where-Object { $_.label -eq ("grc_k{0}_{1}_{2}" -f $r,$p.name,$t) } | Select-Object -First 1
      $relativeRows += [pscustomobject]@{
        prompt=$p.name; tokens=$t; rank=$r;
        baseline_decode_tps=$b.decode_tps; grc_decode_tps=$g.decode_tps;
        decode_pct_of_baseline= if($b.decode_tps){100.0*$g.decode_tps/$b.decode_tps}else{$null};
        baseline_overall_tps=$b.overall_tps; grc_overall_tps=$g.overall_tps;
        overall_pct_of_baseline= if($b.overall_tps){100.0*$g.overall_tps/$b.overall_tps}else{$null};
        baseline_prefill_ms=$b.prefill_ms; grc_prefill_ms=$g.prefill_ms;
        prefill_pct_of_baseline= if($b.prefill_ms){100.0*$g.prefill_ms/$b.prefill_ms}else{$null}
      }
    }
  }
}
$rankRelCsv = Join-Path $OUTDIR 'rank_sweep_relative_to_baseline.csv'
$relativeRows | Export-Csv -NoTypeInformation -Path $rankRelCsv
Write-Host "Rank sweep relative: $rankRelCsv"

# Rank-level aggregate
function Mean($arr){ ($arr|Measure-Object -Average).Average }
$rankAgg = $relativeRows | Group-Object rank | ForEach-Object {
  [pscustomobject]@{
    rank=[int]$_.Name;
    mean_decode_pct_of_baseline = Mean ($_.Group.decode_pct_of_baseline);
    mean_overall_pct_of_baseline = Mean ($_.Group.overall_pct_of_baseline);
    mean_prefill_pct_of_baseline = Mean ($_.Group.prefill_pct_of_baseline)
  }
}
$rankAggPath = Join-Path $OUTDIR 'rank_sweep_aggregate.csv'
$rankAgg | Export-Csv -NoTypeInformation -Path $rankAggPath
Write-Host "Rank sweep aggregate: $rankAggPath"

Write-Host "=== RANK SWEEP COMPLETE ==="
foreach($r in ($rankAgg | Sort-Object rank)){
  Write-Host ("RANK {0} decode%={1:N2} overall%={2:N2} prefill%={3:N2}" -f $r.rank,$r.mean_decode_pct_of_baseline,$r.mean_overall_pct_of_baseline,$r.mean_prefill_pct_of_baseline)
}
