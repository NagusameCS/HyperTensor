# HYPER TENSOR CLEANUP SCRIPT
# 1. Reorganize papers into Volume 1 (I-XV) and Volume 2 (XVI-XXII+)
# 2. Remove ALL emojis/special unicode chars
# 3. Remove ALL bold (, **, __) 
# 4. Remove ALL italics (, , *, _)
# Only processes text files: .py .md .html .json .ps1 .sh .bat .txt .csv .css .js

param(
    [string]$RepoRoot = "c:\Users\legom\HyperTensor",
    [switch]$WhatIf = $false
)

$ErrorActionPreference = "Continue"
$script:changed = 0
$script:files = 0

# File extensions to process
$Exts = @("*.py", "*.md", "*.html", "*.json", "*.ps1", "*.sh", "*.bat", "*.txt", "*.csv", "*.css", "*.js")

# Directories to exclude
$ExcludeDirs = @(".git", ".venv", "node_modules", "__pycache__", "benchmarks", "ARXIV_SUBMISSIONS")

function Should-Process-File($path) {
    foreach ($ex in $ExcludeDirs) {
        if ($path -match "[\\/]$ex[\\/]") { return $false }
    }
    return $true
}

# -- EMOJI / SPECIAL CHAR MAP --
$EmojiMap = @{
    # Checkmarks
    '[PASS]' = '[PASS]'; '[ok]' = '[ok]'; '[ok]' = '[ok]'; '[x]' = '[x]'; '[ ]' = '[ ]'; '[ ]' = '[ ]'
    # X marks
    '[fail]' = '[fail]'; '[fail]' = '[fail]'; '[FAIL]' = '[FAIL]'
    # Warning
    'WARNING:' = 'WARNING:'; 'WARNING:' = 'WARNING:'; '!!' = '!!'
    # Misc symbols  
    '[*]' = '[*]'; '*' = '*'; 'o' = 'o'; '>' = '>'; '>' = '>'
    '*' = '*'; 'o' = 'o'; '#' = '#'; '[ ]' = '[ ]'
    # Arrows
    '->' = '->'; '<-' = '<-'; '^' = '^'; 'v' = 'v'
    '>' = '>'; '<' = '<'; '^' = '^'; 'v' = 'v'
    # Math (keep as ASCII equivalents)
    '×' = 'x'; '÷' = '/'
    # Box drawing (replace with ASCII)
    '+' = '+'; '+' = '+'; '+' = '+'; '+' = '+'
    '|' = '|'; '=' = '='; '+' = '+'; '+' = '+'
    '+' = '+'; '+' = '+'; '+' = '+'
    # Other box drawing chars
    '-' = '-'; '=' = '='; '|' = '|'; '|' = '|'
    '+' = '+'; '+' = '+'; '+' = '+'; '+' = '+'
    '+' = '+'; '+' = '+'; '+' = '+'; '+' = '+'; '+' = '+'
    '#' = '#'; '#' = '#'; '#' = '#'; '|' = '|'; '|' = '|'
    '.' = '.'; ':' = ':'; '#' = '#'
    # Emoji
    '' = ''; '' = ''; '' = ''; '' = ''; '' = ''; '' = ''
    '' = ''; '' = ''; '' = ''; '' = ''; '' = ''; '' = ''
    '' = ''; '' = ''; '' = ''; '' = ''; '' = ''; '' = ''
    '' = ''; '' = ''; '' = ''; '' = ''; '+' = '+'; '-' = '-'
    'x' = 'x'; '/' = '/'; '' = ''; '' = ''; '' = ''; '' = ''
    # Misc
    [char]0x2026 = '...'; [char]0x2013 = '--'; [char]0x2014 = '---'
    [char]0x2019 = "'"; [char]0x2018 = "'"; [char]0x201C = '"'; [char]0x201D = '"'
    # Greek (keep, these are math)
    # 'α' through 'ω' --- KEPT
}

# Build the emoji removal regex
$EmojiChars = ($EmojiMap.Keys | ForEach-Object { [regex]::Escape($_) }) -join ''

Write-Host "=== HYPER TENSOR CLEANUP ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot" -ForegroundColor Cyan
if ($WhatIf) { Write-Host "WHAT-IF MODE --- no changes will be written" -ForegroundColor Yellow }
Write-Host ""

# -- STEP 1: Clean files --
$allFiles = Get-ChildItem -Path $RepoRoot -Recurse -Include $Exts -ErrorAction SilentlyContinue | Where-Object { Should-Process-File $_.FullName }

foreach ($file in $allFiles) {
    $path = $file.FullName
    $content = Get-Content $path -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    
    $original = $content
    $changed_this_file = $false
    
    # -- Remove emojis --
    foreach ($key in $EmojiMap.Keys) {
        if ($content.Contains($key)) {
            $content = $content.Replace($key, $EmojiMap[$key])
            $changed_this_file = $true
        }
    }
    
    # -- Remove HTML bold tags --
    if ($content -match ']*>') {
        $content = $content -replace ']*>', ''
        $content = $content -replace '', ''
        $changed_this_file = $true
    }
    if ($content -match '<b[^>]*>') {
        $content = $content -replace '<b[^>]*>', ''
        $content = $content -replace '</b>', ''
        $changed_this_file = $true
    }
    
    # -- Remove HTML italic tags --
    if ($content -match ']*>') {
        $content = $content -replace ']*>', ''
        $content = $content -replace '', ''
        $changed_this_file = $true
    }
    if ($content -match '<i[^>]*>') {
        $content = $content -replace '<i[^>]*>', ''
        $content = $content -replace '', ''
        $changed_this_file = $true
    }
    
    # -- Remove Markdown bold (**text** and __text__) --
    # Only in .md and .html files (not .py --- ** is valid Python syntax)
    if ($file.Extension -match '\.(md|html)$') {
        if ($content -match '\*\*[^*]+\*\*') {
            $content = $content -replace '\*\*([^*]+)\*\*', '$1'
            $changed_this_file = $true
        }
        # Only replace __text__ that isn't __dunder__ (Python names)
        if ($content -match '(?<![a-zA-Z0-9])__([a-zA-Z][a-zA-Z0-9 ]*[a-zA-Z0-9])__(?![a-zA-Z0-9])') {
            $content = $content -replace '(?<![a-zA-Z0-9])__([a-zA-Z][a-zA-Z0-9 ]*[a-zA-Z0-9])__(?![a-zA-Z0-9])', '$1'
            $changed_this_file = $true
        }
        
        # -- Remove Markdown italic (*text*) --
        # Only where * is used for emphasis, not in math/code
        if ($content -match '(?<!\*)\*([^*\s][^*]*[^*\s])\*(?!\*)') {
            $content = $content -replace '(?<!\*)\*([^*\s][^*]*[^*\s])\*(?!\*)', '$1'
            $changed_this_file = $true
        }
    }
    
    if ($content -ne $original) {
        $script:files++
        $script:changed++
        if (-not $WhatIf) {
            Set-Content -Path $path -Value $content -NoNewline -Encoding UTF8
        }
        $relPath = $path.Replace($RepoRoot, '').TrimStart('\')
        Write-Host "  CLEANED: $relPath" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Files processed: $($script:files)" -ForegroundColor Cyan
Write-Host "Files changed:   $($script:changed)" -ForegroundColor Cyan

# -- STEP 2: Verify no emojis remain --
Write-Host ""
Write-Host "=== VERIFICATION: Checking for remaining emojis ===" -ForegroundColor Cyan
$remaining = @()
foreach ($file in $allFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    foreach ($key in $EmojiMap.Keys) {
        if ($content.Contains($key)) {
            $remaining += "$($file.Name): $key"
            break
        }
    }
}
if ($remaining.Count -eq 0) {
    Write-Host "  NO emojis remain." -ForegroundColor Green
} else {
    Write-Host "  $($remaining.Count) files still have emojis:" -ForegroundColor Yellow
    $remaining | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
}

# -- STEP 3: Verify no bold/italics remain --
Write-Host ""
Write-Host "=== VERIFICATION: Checking for remaining bold/italics ===" -ForegroundColor Cyan
$boldRemaining = @()
$italicRemaining = @()
foreach ($file in $allFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    if ($content -match '|\*\*[^*]+\*\*|__[^_]+__') {
        $boldRemaining += $file.Name
    }
    if ($content -match ']*>|<i[^>]*>') {
        $italicRemaining += $file.Name
    }
}
if ($boldRemaining.Count -eq 0) {
    Write-Host "  NO bold formatting remains." -ForegroundColor Green
} else {
    Write-Host "  $($boldRemaining.Count) files still have bold:" -ForegroundColor Yellow
    $boldRemaining | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
}
if ($italicRemaining.Count -eq 0) {
    Write-Host "  NO italic formatting remains." -ForegroundColor Green
} else {
    Write-Host "  $($italicRemaining.Count) files still have italics:" -ForegroundColor Yellow
    $italicRemaining | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
}

Write-Host ""
Write-Host "=== CLEANUP COMPLETE ===" -ForegroundColor Cyan
