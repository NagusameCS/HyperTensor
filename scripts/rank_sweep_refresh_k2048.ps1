# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
# ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
# ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
# ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
# ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
# ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
# :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
# :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
# ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
# :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
# ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
# ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
# :::::::::................................:@@@@@@@@@@%:...............................::::::
# ::::::::..................................*@@@@@@@@@-................................::::::::
# ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
# :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
# :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
# :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
# :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
# :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
# :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
# :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
# :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
# :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
# ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
# ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
# :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
# ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
# :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
# :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
# ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
# ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
# :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
# ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
# ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
# ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

param(
  [string]$OutDir = '.\benchmarks\whitepaper_pack_20260426_191201'
)

$ErrorActionPreference = 'Stop'
$MODEL = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
$EXE = ".\build_host\geodessical.exe"
$OUTDIR = $OutDir

function Parse-Run([string]$stdoutPath){
  $raw = Get-Content -Raw -Path $stdoutPath
  $mDec = [regex]::Match($raw,'Decode-only:\s*prefill\s*([\d.]+)\s*ms,\s*([\d.]+)\s*tok/s')
  $mGd = [regex]::Match($raw,'\[GD\]\s*(\d+)\s+tokens\s+in\s*([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)')
  [pscustomobject]@{
    decode_tps = if($mDec.Success){ [double]$mDec.Groups[2].Value } else { $null }
    prefill_ms = if($mDec.Success){ [double]$mDec.Groups[1].Value } else { $null }
    overall_tps = if($mGd.Success){ [double]$mGd.Groups[3].Value } else { $null }
    generated_tokens = if($mGd.Success){ [int]$mGd.Groups[1].Value } else { $null }
  }
}

function Resolve-RunPath([string]$baseDir, [string]$label) {
  $p1 = Join-Path $baseDir ("{0}.txt" -f $label)
  if (Test-Path $p1) { return $p1 }
  $p2 = Join-Path $baseDir ("{0}_rep1.txt" -f $label)
  if (Test-Path $p2) { return $p2 }
  throw "missing output for label: $label"
}

$prompts = @(
  @{name='coding'; text='Write a Python function that returns prime numbers up to n.'},
  @{name='reasoning'; text='Explain why gradient clipping helps stabilize training in deep networks.'},
  @{name='factual'; text='Summarize how TCP congestion control works in modern networks.'},
  @{name='creative'; text='Write a short sci-fi paragraph about a city powered by ocean tides.'}
)
$tokenList = @(128,256)

foreach($p in $prompts){
  foreach($t in $tokenList){
    $label = "grc_k2048_{0}_{1}" -f $p.name,$t
    $out = Join-Path $OUTDIR ("{0}_rep1.txt" -f $label)
    $err = Join-Path $OUTDIR ("{0}_err.txt" -f $label)
    Write-Host "Re-running $label"
    & $EXE $MODEL --axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca --axex-compress-rank 2048 --temp 0 -p $p.text -n $t 1> $out 2> $err
    if($LASTEXITCODE -ne 0){ throw "failed: $label" }
    $m = Parse-Run $out
    Write-Host ("  decode={0} prefill={1} overall={2}" -f $m.decode_tps,$m.prefill_ms,$m.overall_tps)
  }
}

# Rebuild rank_sweep_raw.csv from existing output files
$rows = @()
$ranks = @(1024,1536,2048)
foreach($p in $prompts){
  foreach($t in $tokenList){
    $bLabel = "baseline_{0}_{1}" -f $p.name,$t
    $bOut = Resolve-RunPath -baseDir $OUTDIR -label $bLabel
    $b = Parse-Run $bOut
    $rows += [pscustomobject]@{label=$bLabel;prompt=$p.name;tokens=$t;rank=$null;decode_tps=$b.decode_tps;overall_tps=$b.overall_tps;prefill_ms=$b.prefill_ms;generated_tokens=$b.generated_tokens}

    foreach($r in $ranks){
      $gLabel = "grc_k{0}_{1}_{2}" -f $r,$p.name,$t
      $gOut = Resolve-RunPath -baseDir $OUTDIR -label $gLabel
      $g = Parse-Run $gOut
      $rows += [pscustomobject]@{label=$gLabel;prompt=$p.name;tokens=$t;rank=$r;decode_tps=$g.decode_tps;overall_tps=$g.overall_tps;prefill_ms=$g.prefill_ms;generated_tokens=$g.generated_tokens}
    }
  }
}
$rankCsv = Join-Path $OUTDIR 'rank_sweep_raw.csv'
$rows | Export-Csv -NoTypeInformation -Path $rankCsv

$relativeRows = @()
foreach($p in $prompts){
  foreach($t in $tokenList){
    $b = $rows | Where-Object { $_.label -eq ("baseline_{0}_{1}" -f $p.name,$t) } | Select-Object -First 1
    foreach($r in $ranks){
      $g = $rows | Where-Object { $_.label -eq ("grc_k{0}_{1}_{2}" -f $r,$p.name,$t) } | Select-Object -First 1
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

Write-Host "Updated: $rankCsv"
Write-Host "Updated: $rankRelCsv"
Write-Host "Updated: $rankAggPath"
$rankAgg | Sort-Object rank | ForEach-Object {
  Write-Host ("RANK {0} decode%={1:N2} overall%={2:N2} prefill%={3:N2}" -f $_.rank,$_.mean_decode_pct_of_baseline,$_.mean_overall_pct_of_baseline,$_.mean_prefill_pct_of_baseline)
}
