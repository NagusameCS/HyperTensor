$p = 'c:\Users\legom\HyperTensor\docs\HANDOFF_TO_PHD.md'
$t = [IO.File]::ReadAllText($p)
$t = $t.Replace('agtv3_results.json','agt_v3_results.json')
[IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
Write-Host 'done'
