$ErrorActionPreference = 'Stop'

# Final round of doc corrections.
$repls = @(
  @('closexiiisafeogdcreativity.py',          'close_xiii_safe_ogd_creativity.py'),
  @('closexivsnipe_collateral.py',            'close_xiv_snipe_collateral.py'),
  @('closexixiiugtnative.py',                 'close_xi_xii_ugt_native.py'),
  @('riemanncomprehensive_verification.json', 'riemann_comprehensive_verification.json'),
  @('riemannadversarial_results.json',        'riemann_adversarial_results.json'),
  @('riemanncomprehensiveverification.json',  'riemann_comprehensive_verification.json'),
  @('riemannadversarialresults.json',         'riemann_adversarial_results.json'),
  @('riemanncomprehensive/',                  'riemann_comprehensive/'),
  @('riemannadversarial/',                    'riemann_adversarial/'),
  @('benchmarks/riemanncomprehensive',        'benchmarks/riemann_comprehensive'),
  @('benchmarks/riemannadversarial',          'benchmarks/riemann_adversarial'),
  @('benchmarks/acmprototype_results.json',   'benchmarks/acm_prototype_results.json'),
  @('scripts/acmprototype.py',                'scripts/acm_prototype.py'),
  @('scripts/agtv3.py',                       'scripts/agt_v3.py'),
  @('scripts/agtscale_ec2.py',                'scripts/agt_scale_ec2.py'),
  @('attnressweepv5/',                        'attnres_sweep_v5/'),
  @('crosshwlocal/',                          'cross_hw_local_fix_20260428_192807/'),
  @('crosshwremote/',                         'cross_hw_remote_pull_20260428_174400/')
)

$files = Get-ChildItem c:\Users\legom\HyperTensor\docs\*.md, c:\Users\legom\HyperTensor\docs\*.html, c:\Users\legom\HyperTensor\REPRODUCTION.md
foreach ($f in $files) {
  $orig = [IO.File]::ReadAllText($f.FullName)
  $t = $orig
  foreach ($pair in $repls) { $t = $t.Replace($pair[0], $pair[1]) }
  if ($t -ne $orig) {
    [IO.File]::WriteAllText($f.FullName, $t, [System.Text.UTF8Encoding]::new($false))
    Write-Host ("FIXED  " + $f.Name + "  (" + ($t.Length - $orig.Length) + " chars)")
  }
}
Write-Host 'Done.'
