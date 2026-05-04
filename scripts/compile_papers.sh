#!/bin/bash
cd /tmp/compile
for f in ugt_taxonomy geodesic_compiler geodesic_synthesis geodesic_sniping organic_generation; do
    echo "=== Compiling $f ==="
    pdflatex -interaction=nonstopmode ${f}.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode ${f}.tex > /dev/null 2>&1
    if [ -f ${f}.pdf ]; then
        SIZE=$(ls -la ${f}.pdf | awk '{print $5}')
        echo "  OK: ${f}.pdf ($SIZE bytes)"
    else
        echo "  FAILED: ${f}.pdf not created"
    fi
done
echo "DONE"
