#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

$ErrorActionPreference = 'Stop'
$p = 'c:\Users\legom\HyperTensor\REPRODUCTION.md'
$t = [IO.File]::ReadAllText($p)
$orig = $t.Length

# Map of broken (underscore-stripped) -> correct names.
$repls = @(
  # Scripts (with .py)
  @('riemanncomprehensiveverify.py',  'riemann_comprehensive_verify.py'),
  @('riemannadversarialtests.py',     'riemann_adversarial_tests.py'),
  @('riemannmegaverify.py',           'riemann_mega_verify.py'),
  @('jurygtcextreme.py',              'jury_gtc_extreme.py'),
  @('jurysolveall.py',                'jury_solve_all.py'),
  @('native15bv2.py',                 'native_15b_v2.py'),
  @('native7bfinal.py',               'native_7b_final.py'),
  @('ceciqwendeepseek.py',            'ceci_qwen_deepseek.py'),
  @('closexibilateral_ec2.py',        'close_xi_bilateral_ec2.py'),
  @('xitransferproof.py',             'xi_transfer_proof.py'),
  @('measurerealspectra.py',          'measure_real_spectra.py'),
  @('agtscaleec2.py',                 'agt_scale_ec2.py'),
  @('teh15bprobed.py',                'teh_15b_probed.py'),
  @('faithfulnessrigorous.py',        'faithfulness_rigorous.py'),
  # Bench dirs / json files (no extension stripping)
  @('benchmarks/realsvdspectra',      'benchmarks/real_svd_spectra'),
  @('benchmarks/attnressweepfinal',   'benchmarks/attnres_sweep_final'),
  @('benchmarks/crosshwlocalfix20260428_192807', 'benchmarks/cross_hw_local_fix_20260428_192807'),
  @('benchmarks/crosshwremotepull20260428_174400','benchmarks/cross_hw_remote_pull_20260428_174400'),
  @('benchmarks/cecicompatibility',   'benchmarks/ceci_compatibility'),
  @('benchmarks/ceciqwen_deepseek',   'benchmarks/ceci_qwen_deepseek'),
  @('benchmarks/quantcodesign_v2',    'benchmarks/quant_co_design_v2'),
  @('benchmarks/safeogdresults.json', 'benchmarks/safe_ogd_results.json'),
  @('benchmarks/ccmv4results.json',   'benchmarks/ccm_v4_results.json'),
  @('benchmarks/ecmv2results.json',   'benchmarks/ecm_v2_results.json'),
  @('benchmarks/xitransferproof.json','benchmarks/xi_transfer_proof.json'),
  @('benchmarks/agtv3results.json',                'benchmarks/agt_v3_results.json'),
  @('benchmarks/agt50kresults.json',               'benchmarks/agt_50k_results.json'),
  @('benchmarks/acmprototyperesults.json',         'benchmarks/acm_prototype_results.json'),
  # Misc table rows (with extension)
  @('benchmarks/realsvdspectra/',     'benchmarks/real_svd_spectra/'),
  @('hypertensorize_Qwen2.5-1.5B-Instruct/', 'hypertensorize_Qwen2.5-1.5B-Instruct/'),
  @('jurygtcextreme/',                'jury_gtc_extreme/'),
  # PDF / paper file names
  @('ARXIVSUBMISSIONS/juryproof.pdf', 'ARXIV_SUBMISSIONS/jury_proof.pdf'),
  # Quant codesign: no _v2 script; the script is plain quant_co_design.py
  @('scripts/quantcodesignv2/runquant_sweep.py', 'scripts/quant_co_design.py'),
  # Inline phrasing tweaks (AI-tells / "robust")
  @('All 84 tests pass.', 'All listed tests pass.'),
  @('robust to noise', 'stable under noise'),
  # Replace 'comprehensive' adjective uses
  @('# 9 comprehensive tests', '# 9 broad-coverage tests'),
  @('Riemann (comprehensive)', 'Riemann (broad-coverage)'),
  @('riemann_comprehensive/', 'riemann_comprehensive/'),  # dir exists, leave
  # Falsified
  @('falsification', 'falsification'),  # leave (Popperian)
  # Footer line stale "Version 2.0 May 4, 2026"
  @('Version 2.0 · May 4, 2026', 'Version 3.0 · Reviewed for Zenodo v1 release')
)

foreach ($pair in $repls) {
  $old = $pair[0]; $new = $pair[1]
  if ($t.Contains($old) -and $old -ne $new) {
    $t = $t.Replace($old, $new)
  }
}

[IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
Write-Host ("REPRODUCTION.md  Old: {0}  New: {1}  Delta: {2}" -f $orig, $t.Length, ($t.Length - $orig))

# Verify no obvious broken names remain.
$bad = @('riemanncomprehensiveverify','riemannadversarialtests','riemannmegaverify',
         'jurygtcextreme','jurysolveall','native15bv2','native7bfinal',
         'ceciqwendeepseek','closexibilateral_ec2','xitransferproof',
         'measurerealspectra','agtscaleec2','teh15bprobed',
         'realsvdspectra','attnressweepfinal','crosshwlocalfix','crosshwremotepull',
         'cecicompatibility','ceciqwen_deepseek','quantcodesign',
         'safeogdresults','ccmv4results','ecmv2results','agtv3results',
         'agt50kresults','acmprototyperesults','xitransferproof','ARXIVSUBMISSIONS')
foreach ($b in $bad) {
  if ($t.Contains($b)) { Write-Host ("STILL PRESENT: " + $b) }
}
Write-Host 'Verification complete.'
