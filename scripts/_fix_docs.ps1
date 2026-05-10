# Apply file-name corrections (stripped underscores) to all docs/*.md and docs/*.html.
$ErrorActionPreference = 'Stop'

# Map of broken -> correct.
$repls = @(
  @('riemanncomprehensiveverify.py', 'riemann_comprehensive_verify.py'),
  @('riemannadversarialtests.py',    'riemann_adversarial_tests.py'),
  @('riemannmegaverify.py',          'riemann_mega_verify.py'),
  @('jurygtcextreme.py',             'jury_gtc_extreme.py'),
  @('jurysolveall.py',               'jury_solve_all.py'),
  @('native15bv2.py',                'native_15b_v2.py'),
  @('native7bfinal.py',              'native_7b_final.py'),
  @('ceciqwendeepseek.py',           'ceci_qwen_deepseek.py'),
  @('closexibilateral_ec2.py',       'close_xi_bilateral_ec2.py'),
  @('xitransferproof.py',            'xi_transfer_proof.py'),
  @('measurerealspectra.py',         'measure_real_spectra.py'),
  @('agtscaleec2.py',                'agt_scale_ec2.py'),
  @('teh15bprobed.py',               'teh_15b_probed.py'),
  @('teh15bprobed_results.json',     'teh_15b_probed_results.json'),
  @('faithfulnessrigorous.py',       'faithfulness_rigorous.py'),
  # bench dirs/files
  @('realsvdspectra',                'real_svd_spectra'),
  @('attnressweepfinal',             'attnres_sweep_final'),
  @('crosshwlocalfix',               'cross_hw_local_fix_'),
  @('crosshwremotepull',             'cross_hw_remote_pull_'),
  @('cecicompatibility',             'ceci_compatibility'),
  @('ceciqwen_deepseek',             'ceci_qwen_deepseek'),
  @('quantcodesign_v2',              'quant_co_design_v2'),
  @('safeogdresults.json',           'safe_ogd_results.json'),
  @('ccmv4results.json',             'ccm_v4_results.json'),
  @('ecmv2results.json',             'ecm_v2_results.json'),
  @('xitransferproof.json',          'xi_transfer_proof.json'),
  @('agtv3results.json',             'agt_v3_results.json'),
  @('agt50kresults.json',            'agt_50k_results.json'),
  @('acmprototyperesults.json',      'acm_prototype_results.json'),
  @('ARXIVSUBMISSIONS',              'ARXIV_SUBMISSIONS'),
  @('juryproof.pdf',                 'jury_proof.pdf'),
  @('juryproof.tex',                 'jury_proof.tex')
)

# But: my crosshwlocalfix -> cross_hw_local_fix_ would double-apply if a date follows.
# So explicitly fix the two real names instead:
$datedRepls = @(
  @('crosshwlocalfix20260428_192807',  'cross_hw_local_fix_20260428_192807'),
  @('crosshwremotepull20260428_174400','cross_hw_remote_pull_20260428_174400')
)

$files = Get-ChildItem c:\Users\legom\HyperTensor\docs\*.md, c:\Users\legom\HyperTensor\docs\*.html

foreach ($f in $files) {
  $orig = [IO.File]::ReadAllText($f.FullName)
  $t = $orig
  foreach ($pair in $datedRepls) { $t = $t.Replace($pair[0], $pair[1]) }
  foreach ($pair in $repls) {
    if ($pair[0] -in @('crosshwlocalfix','crosshwremotepull')) { continue }
    $t = $t.Replace($pair[0], $pair[1])
  }
  if ($t -ne $orig) {
    [IO.File]::WriteAllText($f.FullName, $t, [System.Text.UTF8Encoding]::new($false))
    Write-Host ("FIXED  " + $f.Name + "  (" + ($t.Length - $orig.Length) + " chars)")
  } else {
    Write-Host ("clean  " + $f.Name)
  }
}
