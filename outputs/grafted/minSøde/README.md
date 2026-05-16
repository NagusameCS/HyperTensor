<!-- :%#@=+-.                              .-=+@#%: -->
<!--  .:+#%@@@@@@@@@@@@@@@@@@@@@@@@#=...:=#@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@%-.:+%@@@@@#=:=%@@@@@@@@@@@@@@@@@%+-. -->
<!--      :-%@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@@@%=-@@@@@@@@@@@@@@@. -->
<!--         :+#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+=*@@@@@@@@@@@#: -->
<!--            -*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@+. -->
<!--              .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@*. -->
<!--                .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--                  -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                   :%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%: -->
<!--                    .*@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--                     :#@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                      .#@@@@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@+ -->
<!--                        .@@@@@@@@%+-::=@@@@@@@@: -->
<!--                        .*@@@@@@@.       .+@@@@@@: -->
<!--                        .*@@@@@@@:         :@@@@@@: -->
<!--                        .*@@@@@@@%+-::::-+#@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   HyperTensor -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   universal geometric tensor framework -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   Papers I-XXX -->
<!--                        :#@@@@@@@@@@@@@@@@@@@@@@@*.   15/18 at 100% -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@%.   Jury-GTC @ 53x -->
<!--                      .+@@@@@@@@@@@@@@@@@@@@@@@@@@#:   AGT @ 50K primes -->
<!--                     -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.   External verify 14/14 -->
<!--                   :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+.   COG 10K converged -->
<!--                 .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.  Bilateral 1.5B 0.968 -->
<!--               :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--            .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--         .-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@. -->
<!--      .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+: -->
<!-- .:+#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->

---
language: en
tags:
- hypertensor
- ceci-graft
- danish
- smollm2
- experimental
pipeline_tag: text-generation
license: apache-2.0
---

# minSøde (my sweet)

Layer 5 attention chimaera. Early-layer FFN transplanted via GRC basis alignment from donor layer 15. 105% PPL recovery — grafted model outperforms original on simple sentences. The FFN from a deep reasoning layer now processes at an early attention layer, creating a unique 'deep insight at shallow depth' effect.

## Architecture

- Base: SmolLM2-135M-Instruct
- Method: CECI Protocol (HyperTensor Paper X) — GRC basis projection
- Created: 2026-05-04
- Repository: [HyperTensor](https://github.com/NagusameCS/HyperTensor)

## Graft Proof

This model was created by:
1. Computing the GRC (Geodesic Residual Compression) basis from the target layer's attention weights via SVD
2. Projecting the donor layer's FFN weights into the target's geometric subspace
3. Blending at controlled strength to preserve stability

Perplexity testing confirms the graft transfers functional structure without destroying the model.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.frompretrained("NagusameCS/minSøde", trustremote_code=True)
tokenizer = AutoTokenizer.from_pretrained("NagusameCS/minSøde")
```
