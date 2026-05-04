## Paper B --- Load & VRAM Efficiency

Summary: load logs show several compressed models fitting within an 8 GB-class GPU budget, with model-dependent offload behavior.

| Model | Load (ms) | GPU tensors | GPU VRAM (MB) | Offload from | tok/s |
|-------|----------|-------------|---------------|--------------|-------|
| GLM-4.7-Flash | 817 | 5 | 457 | --- | --- |
| Gemma3-12B | 5400 | 236 | 6019 | 47 | 3.8 |
| Gemma3-4B | 2280 | 171 | 2174 | --- | 58.3 |
| Gemma4-31B | 6155 | 103 | 6444 | 21 | --- |
| Qwen3.5-35B | 1220 | 82 | 1051 | --- | --- |
