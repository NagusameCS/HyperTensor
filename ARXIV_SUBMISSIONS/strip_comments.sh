#!/usr/bin/env bash
# strip_comments.sh --- remove TeX comments before arXiv upload.
#
# arXiv archives all uploaded source forever. This strips any line-comment
# content (text after an unescaped %) and any wholly-commented lines.
#
# Usage:   ./strip_comments.sh main.tex > main.cleaned.tex
#
# Behaviour:
#   - Preserves \%  (escaped percent) untouched.
#   - Removes inline comments after an unescaped %.
#   - Removes lines that are entirely comments.
#   - Preserves verbatim/listings blocks (best-effort: skip blocks between
#     \begin{lstlisting}/\end{lstlisting} and \begin{verbatim}/\end{verbatim}).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 file.tex" >&2
    exit 1
fi

awk '
BEGIN { in_verb = 0 }
/\\begin\{(lstlisting|verbatim|minted)/ { in_verb = 1; print; next }
/\\end\{(lstlisting|verbatim|minted)/   { in_verb = 0; print; next }
{
    if (in_verb) { print; next }
    line = $0
    out = ""
    i = 1
    n = length(line)
    while (i <= n) {
        c = substr(line, i, 1)
        if (c == "\\" && i < n) {
            out = out c substr(line, i+1, 1)
            i += 2
            continue
        }
        if (c == "%") break
        out = out c
        i += 1
    }
    # drop wholly-commented lines (out is empty/whitespace and the original
    # line started with %)
    if (out ~ /^[[:space:]]*$/ && line ~ /^[[:space:]]*%/) next
    sub(/[[:space:]]+$/, "", out)
    print out
}
' "$1"
