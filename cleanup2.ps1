# HYPER TENSOR CLEANUP --- emoji, bold, italics removal
param([string]$RepoRoot = "c:\Users\legom\HyperTensor", [switch]$WhatIf)

$ErrorActionPreference = "Continue"
$changed = 0
$files = 0

# -- EMOJI REPLACEMENTS (find, replace) --
$emojiReplacements = @(
    @('[PASS]','[PASS]'), @('[ok]','[ok]'), @('[ok]','[ok]'), @('[fail]','[fail]'), @('[fail]','[fail]'), @('[FAIL]','[FAIL]'),
    @('[x]','[x]'), @('[ ]','[ ]'), @('[ ]','[ ]'),
    @('WARNING:','WARNING:'), @('WARNING:','WARNING:'), @('!!','!!'), @('[*]','[*]'),
    @('*','*'), @('o','o'), @('>','>'), @('>','>'), @('*','*'), @('o','o'),
    @('#','#'), @('[ ]','[ ]'),
    @('->','->'), @('<-','<-'), @('^','^'), @('v','v'), @('>','>'), @('<','<'),
    @('^','^'), @('v','v'),
    @('+','+'), @('+','+'), @('+','+'), @('+','+'), @('|','|'), @('=','='),
    @('+','+'), @('+','+'), @('+','+'), @('+','+'), @('+','+'),
    @('-','-'), @('=','='), @('|','|'), @('|','|'),
    @('+','+'), @('+','+'), @('+','+'), @('+','+'),
    @('+','+'), @('+','+'), @('+','+'), @('+','+'), @('+','+'),
    @('#','#'), @('#','#'), @('#','#'), @('|','|'), @('|','|'),
    @('.','.'), @(':',':'), @('#','#'),
    @('',''), @('',''), @('',''), @('',''), @('',''), @('',''),
    @('',''), @('',''), @('',''), @('',''), @('',''), @('',''),
    @('',''), @('',''), @('',''), @('',''), @('',''), @('',''),
    @('',''), @('',''), @('',''), @('',''),
    @('+','+'), @('-','-'), @('x','x'), @('/','/'),
    @('',''), @('',''), @('',''), @('',''),
    @('...','...'), @('--','--'), @('---','---'),
    @([char]0x2019,"'"), @([char]0x2018,"'"), @([char]0x201C,'"'), @([char]0x201D,'"')
)

$Exts = @("*.py", "*.md", "*.html", "*.json", "*.ps1", "*.sh", "*.bat", "*.txt", "*.csv", "*.css", "*.js")
$ExcludeDirs = @(".git", ".venv", "node_modules", "__pycache__")

function ShouldProcess($p) {
    foreach ($ex in $ExcludeDirs) { if ($p -match "[\\/]$ex[\\/]") { return $false } }
    return $true
}

Write-Host "=== HYPER TENSOR CLEANUP ===" -ForegroundColor Cyan
if ($WhatIf) { Write-Host "WHAT-IF MODE" -ForegroundColor Yellow }
Write-Host ""

$allFiles = Get-ChildItem -Path $RepoRoot -Recurse -Include $Exts -ErrorAction SilentlyContinue |
    Where-Object { ShouldProcess $_.FullName }

foreach ($file in $allFiles) {
    $path = $file.FullName
    $content = Get-Content $path -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    $original = $content
    $changedThis = $false

    # 1. Replace emojis
    foreach ($pair in $emojiReplacements) {
        $find = $pair[0]; $replace = $pair[1]
        if ($content.Contains($find)) {
            $content = $content.Replace($find, $replace)
            $changedThis = $true
        }
    }

    # 2. Remove HTML bold tags
    if ($content -match ']*>', ''
        $content = $content -replace '', ''
        $changedThis = $true
    }
    if ($content -match '') {
        $content = $content -replace '', ''
        $content = $content -replace '', ''
        $changedThis = $true
    }

    # 3. Remove HTML italic tags
    if ($content -match ']*>', ''
        $content = $content -replace '', ''
        $changedThis = $true
    }
    if ($content -match '') {
        $content = $content -replace '', ''
        $content = $content -replace '', ''
        $changedThis = $true
    }

    # 4. Remove Markdown bold/italic (only in .md and .html files --- not .py code)
    if ($file.Extension -match '\.(md|html)$') {
        if ($content -match '\*\*[^*]+\*\*') {
            $content = $content -replace '\*\*([^*]+)\*\*', '$1'
            $changedThis = $true
        }
        if ($content -match '(?<!\*)\*([^*\s][^*]{0,80}?[^*\s])\*(?!\*)') {
            $content = $content -replace '(?<!\*)\*([^*\s][^*]{0,80}?[^*\s])\*(?!\*)', '$1'
            $changedThis = $true
        }
    }

    if ($content -ne $original) {
        $files++
        if (-not $WhatIf) {
            [System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))
        }
        $rel = $path.Replace($RepoRoot, '').TrimStart('\')
        Write-Host "  CLEANED: $rel" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Files scanned:  $($allFiles.Count)" -ForegroundColor Cyan
Write-Host "Files changed:  $files" -ForegroundColor Cyan
Write-Host "=== DONE ===" -ForegroundColor Cyan
