#!/bin/bash
set -e
rm -f /root/models/ott_*.bin /root/models/ott_*.dat /root/models/axiom_*.bin
/root/geodessical /root/models/llama31-8b-q8_0.gguf \
    --axiom-beta-run \
    --axex-compress \
    --axiom-skip-geodesic \
    -p "Hello, my name is" \
    -n 80 2>&1
