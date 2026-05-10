$p = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex'
$t = [IO.File]::ReadAllText($p)
$ai_tells = @(
  '\bdelve\b',
  '\bleverag',
  '\bcomprehensive',
  '\brobust',
  '\bnovel\b',
  '\butili[sz]e\b',
  '\bplethora\b',
  '\bmyriad\b',
  '\bunderpin',
  '\bharness',
  '\bunlock',
  '\bseamless',
  '\bcutting[- ]edge',
  '\bstate[- ]of[- ]the[- ]art',
  '\bsynerg',
  '\bholistic',
  '\bjourney\b',
  '\bmeticulous',
  '\bquintessential',
  '\bexemplifies',
  '\bepitomi',
  '\bstands as',
  '\bplays a (?:pivotal|crucial|key) role',
  '\bIn (?:summary|conclusion|essence)',
  '\bAs we (?:can|have) (?:see|seen)',
  '\b(?:Crucially|Notably|Importantly|Furthermore|Moreover),',
  'sheds light on',
  'a deep dive'
)
foreach ($pat in $ai_tells) {
  $c = ([regex]::Matches($t, $pat, 'IgnoreCase')).Count
  if ($c -gt 0) {
    Write-Host ("{0,4}  {1}" -f $c, $pat)
  }
}
Write-Host ""
Write-Host ("ASCII em-dash sequences ---:  " + ([regex]::Matches($t, '---')).Count)
