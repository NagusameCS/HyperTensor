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

# Build Zenodo upload metadata from refs.bib + the volume.
#   - zenodo_related_identifiers.tsv  (Relation | Identifier | Scheme | Resource type)
#   - zenodo_references.txt           (one formatted reference per line)

$ErrorActionPreference = 'Stop'
$bibPath = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\refs.bib'
$bib = [IO.File]::ReadAllText($bibPath)

# Split into entries. Each entry starts with @TYPE{key, ... } at column 0.
$entries = [regex]::Matches($bib, '(?ms)^@(\w+)\{([^,]+),(.*?)\n\}\s*')

function Get-Field {
  param([string]$body, [string]$name)
  $m = [regex]::Match($body, '(?im)^\s*' + $name + '\s*=\s*(\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)")')
  if ($m.Success) {
    if ($m.Groups[2].Success) { return $m.Groups[2].Value.Trim() }
    return $m.Groups[3].Value.Trim()
  }
  return ''
}

function Clean-Tex {
  param([string]$s)
  $t = $s
  $t = $t -replace '\\&', '&'
  $t = $t -replace '\\textit\{([^{}]*)\}', '$1'
  $t = $t -replace '\\textbf\{([^{}]*)\}', '$1'
  $t = $t -replace '\\emph\{([^{}]*)\}', '$1'
  $t = $t -replace '\\texttt\{([^{}]*)\}', '$1'
  $t = $t -replace '\\url\{([^{}]*)\}', '$1'
  # Accents: \'{e} \"{o} \^{a} \`{e} \~{n} \={a}  -> base letter
  $t = [regex]::Replace($t, '\\[''"\^~=`]\s*\{?([A-Za-z])\}?', '$1')
  $t = $t -replace "\\\{L\}", 'L'
  $t = $t -replace "\\\{l\}", 'l'
  $t = $t -replace "\\.", ''
  $t = $t -replace '~', ' '
  $t = $t -replace '--', "-"
  $t = $t -replace '\{', ''
  $t = $t -replace '\}', ''
  $t = $t -replace '\s+', ' '
  return $t.Trim().TrimEnd(',').TrimEnd('.')
}

function Format-Authors {
  param([string]$raw)
  if (-not $raw) { return '' }
  # Strip TeX accents in the raw author string before splitting.
  $raw = [regex]::Replace($raw, '\\[''"\^~=`]\s*\{?([A-Za-z])\}?', '$1')
  $raw = $raw -replace '\{', '' -replace '\}', ''
  $parts = $raw -split '\s+and\s+'
  $out = New-Object System.Collections.Generic.List[string]
  foreach ($p in $parts) {
    $p = $p.Trim()
    if (-not $p) { continue }
    if ($p -match ',') {
      $bits = $p -split ',', 2
      $family = $bits[0].Trim()
      $given  = $bits[1].Trim()
      $initials = ($given -split '\s+|-' | Where-Object { $_ } | ForEach-Object { $_.Substring(0,1).ToUpper() + '.' }) -join ' '
      $out.Add("$family, $initials")
    } else {
      $bits = $p -split '\s+'
      if ($bits.Count -ge 2) {
        $family = $bits[-1]
        $given  = $bits[0..($bits.Count-2)] -join ' '
        $initials = ($given -split '\s+|-' | Where-Object { $_ } | ForEach-Object { $_.Substring(0,1).ToUpper() + '.' }) -join ' '
        $out.Add("$family, $initials")
      } else {
        $out.Add($p)
      }
    }
  }
  if ($out.Count -le 6) { return ($out -join ', ') }
  return ($out[0..5] -join ', ') + ', et al'
}

# Build outputs.
$relRows = New-Object System.Collections.Generic.List[string]
$relRows.Add("Relation`tIdentifier`tScheme`tResource type")
$seenIds = New-Object System.Collections.Generic.HashSet[string]

$refLines = New-Object System.Collections.Generic.List[string]
$refLines.Add("# References (paste into Zenodo 'References' textbox)")
$refLines.Add("")

foreach ($e in $entries) {
  $type = $e.Groups[1].Value
  $key  = $e.Groups[2].Value.Trim()
  $body = $e.Groups[3].Value

  $title    = Clean-Tex (Get-Field $body 'title')
  $author   = Get-Field $body 'author'
  $year     = Get-Field $body 'year'
  $eprint   = Get-Field $body 'eprint'
  $doi      = Get-Field $body 'doi'
  $url      = Get-Field $body 'url'
  $how      = Get-Field $body 'howpublished'
  $journal  = Clean-Tex (Get-Field $body 'journal')
  $book     = Clean-Tex (Get-Field $body 'booktitle')
  $publisher= Clean-Tex (Get-Field $body 'publisher')
  $note     = Clean-Tex (Get-Field $body 'note')

  # Try to recover bare arXiv ID from howpublished/note/journal if eprint missing.
  if (-not $eprint) {
    foreach ($f in @($how, $note, $journal, $url)) {
      if (-not $f) { continue }
      $m = [regex]::Match($f, '(?:arXiv\s*:?|abs/)\s*(\d{4}\.\d{4,5})', 'IgnoreCase')
      if ($m.Success) { $eprint = $m.Groups[1].Value; break }
    }
  }

  $authorsFmt = Format-Authors $author

  # Build human-readable reference string.
  $venue = ''
  if     ($journal) { $venue = $journal }
  elseif ($book)    { $venue = $book }
  elseif ($publisher) { $venue = $publisher }
  elseif ($how)     { $venue = (Clean-Tex $how) }
  elseif ($note)    { $venue = $note }

  $idTail = ''
  if     ($eprint) { $idTail = "arXiv:$eprint" }
  elseif ($doi)    { $idTail = "doi:$doi" }
  elseif ($url)    { $idTail = $url }

  $refStr = ''
  if ($authorsFmt) { $refStr += $authorsFmt + '. ' }
  if ($title)      { $refStr += $title + '. ' }
  if ($venue)      { $refStr += $venue + ', ' }
  if ($year)       { $refStr += $year + '. ' }
  if ($idTail)     { $refStr += $idTail + '.' }
  $refStr = ($refStr -replace '\s+', ' ').Trim()
  if ($refStr) { $refLines.Add($refStr) }

  # Add to related identifiers if it has a structured ID.
  if ($eprint) {
    if ($seenIds.Add("arXiv:$eprint")) {
      $relRows.Add("References`tarXiv:$eprint`tarXiv`tPublication / Preprint")
    }
  } elseif ($doi) {
    if ($seenIds.Add("doi:$doi")) {
      $relRows.Add("References`thttps://doi.org/$doi`tDOI`tPublication")
    }
  }
}

# Add the GitHub repo as supplemental software.
$relRows.Add("Is supplemented by`thttps://github.com/NagusameCS/HyperTensor`tURL`tSoftware")

# Write outputs.
$relPath = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\zenodo_v1\zenodo_related_identifiers.tsv'
$refPath = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\zenodo_v1\zenodo_references.txt'
[IO.File]::WriteAllText($relPath, ($relRows -join "`r`n") + "`r`n", [System.Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($refPath, ($refLines -join "`r`n") + "`r`n", [System.Text.UTF8Encoding]::new($false))

Write-Host ("Wrote " + $relPath + " (" + $relRows.Count + " rows incl. header)")
Write-Host ("Wrote " + $refPath + " (" + $refLines.Count + " lines incl. header)")
