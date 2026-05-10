$p = 'c:\Users\legom\HyperTensor\REPRODUCTION.md'
$t = [IO.File]::ReadAllText($p)
$t = $t.Replace('  agtv3results.json',          '  agt_v3_results.json')
$t = $t.Replace('  agt50kresults.json',         '  agt_50k_results.json')
$t = $t.Replace('  acmprototyperesults.json',   '  acm_prototype_results.json')
$t = $t.Replace('  realsvdspectra/',            '  real_svd_spectra/')
[IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
Write-Host 'Done.'
