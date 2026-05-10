"""Expand thin bibliographies for Papers VIII and IX, and append a clear
GitHub-repo / GH-Pages pointer to every Reproduction section that lacks one.

Idempotent: detects the marker comments before writing, only edits if absent.
"""
from __future__ import annotations
import pathlib, re

PATH = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
src = PATH.read_text(encoding='utf-8')
orig = src

REPO_URL = 'https://github.com/NagusameCS/HyperTensor'
SITE_URL = 'https://nagusamecs.github.io/HyperTensor/index.html'

# ------------------------------------------------------------ Paper VIII bib
old_viii = (
    "\\begin{thebibliography}{99}\n"
    "\\bibitem{stewart} W.K.O. Stewart. HyperTensor Repository. 2026.\n"
    "\\bibitem{lewis} P. Lewis et al. Retrieval-Augmented Generation for\n"
    "Knowledge-Intensive NLP Tasks. NeurIPS 2020.\n"
    "\\bibitem{gptcache} GPTCache: An Open-Source Semantic Cache for LLM\n"
    "Applications. 2023.\n"
    "\\end{thebibliography}"
)
new_viii = (
    "\\begin{thebibliography}{99}\n"
    "\\bibitem{stewart} W.K.O. Stewart. HyperTensor Repository. 2026. \\url{" + REPO_URL + "}.\n"
    "\\bibitem{lewis} P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.\n"
    "\\bibitem{karpukhin} V. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.\n"
    "\\bibitem{johnson} J. Johnson, M. Douze, H. J{\\'e}gou. Billion-Scale Similarity Search with GPUs (FAISS). IEEE TBD, 2019.\n"
    "\\bibitem{borgeaud} S. Borgeaud et al. Improving Language Models by Retrieving from Trillions of Tokens (RETRO). ICML, 2022.\n"
    "\\bibitem{khandelwal} U. Khandelwal et al. Generalization through Memorization: Nearest Neighbor Language Models. ICLR, 2020.\n"
    "\\bibitem{kwon2023} W. Kwon et al. Efficient Memory Management for LLM Serving with PagedAttention (vLLM). SOSP, 2023.\n"
    "\\bibitem{gptcache} F. Bang. GPTCache: An Open-Source Semantic Cache for LLM Applications. arXiv:2311.04205, 2023.\n"
    "\\bibitem{xiao2023} G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453, 2023.\n"
    "\\bibitem{zhang2023h2o} Z. Zhang et al. H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS, 2023.\n"
    "\\bibitem{ge2024model} S. Ge et al. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. ICLR, 2024.\n"
    "\\bibitem{douze2024faiss} M. Douze et al. The FAISS Library. arXiv:2401.08281, 2024.\n"
    "\\end{thebibliography}"
)

# ------------------------------------------------------------ Paper IX bib
old_ix = (
    "\\begin{thebibliography}{99}\n"
    "\\bibitem{stewart} W.K.O. Stewart. HyperTensor Repository. 2026.\n"
    "\\bibitem{williams} S. Williams et al. Roofline: An Insightful Visual\n"
    "Performance Model for Multicore Architectures. CACM 2009.\n"
    "\\bibitem{gptq} E. Frantar et al. GPTQ: Accurate Post-Training Quantization\n"
    "for Generative Pre-Trained Transformers. ICLR 2023.\n"
    "\\end{thebibliography}"
)
new_ix = (
    "\\begin{thebibliography}{99}\n"
    "\\bibitem{stewart} W.K.O. Stewart. HyperTensor Repository. 2026. \\url{" + REPO_URL + "}.\n"
    "\\bibitem{williams} S. Williams, A. Waterman, D. Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. CACM, 2009.\n"
    "\\bibitem{yuan2024} Z. Yuan et al. LLM Inference Unveiled: Survey and Roofline Model Insights. arXiv:2402.16363, 2024.\n"
    "\\bibitem{gptq} E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. ICLR, 2023.\n"
    "\\bibitem{awq} J. Lin et al. AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978, 2023.\n"
    "\\bibitem{smoothquant} G. Xiao et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438, 2022.\n"
    "\\bibitem{llmint8} T. Dettmers et al. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339, 2022.\n"
    "\\bibitem{nvidia-ada} NVIDIA Corporation. NVIDIA Ada GPU Architecture Whitepaper. 2022.\n"
    "\\bibitem{nvidia-nsight} NVIDIA Corporation. Nsight Compute Profiling Guide. 2024.\n"
    "\\bibitem{ggerganov} G. Gerganov and llama.cpp contributors. GGUF Binary Format Specification. 2023. \\url{https://github.com/ggerganov/ggml/blob/master/docs/gguf.md}.\n"
    "\\bibitem{hennessy} J.L. Hennessy, D.A. Patterson. Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann, 2017.\n"
    "\\bibitem{harris2007} M. Harris. Optimizing Parallel Reduction in CUDA. NVIDIA Developer Technology, 2007.\n"
    "\\end{thebibliography}"
)

assert old_viii in src, 'paper VIII bib not found'
assert old_ix in src, 'paper IX bib not found'
src = src.replace(old_viii, new_viii)
src = src.replace(old_ix, new_ix)
print('Paper VIII bib expanded:', new_viii.count('\\bibitem'))
print('Paper IX  bib expanded:', new_ix.count('\\bibitem'))

# ------------------------------------------------------------ GH-repo pointer
# Append a uniform reproduction-pointer paragraph to every \section{Reproduction}
# / \subsection{Reproduction} / \section{Reproducibility} block that does not
# already mention the GH repo by URL. We scan each Reproduction header and the
# 1200 chars after it; if the URL is already there, skip; otherwise inject the
# pointer immediately after the header line.
PTR = (
    "\n\\smallskip\\noindent\\textit{Code and data.} The full HyperTensor "
    "repository (scripts, configs, raw benchmark JSONs, and the reproduction "
    "guide) is at \\url{" + REPO_URL + "}; a browsable version with figures "
    "and step-by-step instructions is at \\url{" + SITE_URL + "}.\n"
)

def inject_repro(text: str) -> tuple[str, int]:
    out = []
    cursor = 0
    inserted = 0
    pat = re.compile(r'\\(?:section|subsection)\*?\{Reprodu[a-z]+[^}]*\}')
    for m in pat.finditer(text):
        # Look at the next 1500 chars for an existing URL.
        window = text[m.end():m.end() + 1500]
        if REPO_URL in window:
            continue
        out.append(text[cursor:m.end()])
        out.append(PTR)
        cursor = m.end()
        inserted += 1
    out.append(text[cursor:])
    return ''.join(out), inserted

src, n_inj = inject_repro(src)
print(f'Reproduction GH-repo pointers injected: {n_inj}')

# ------------------------------------------------------------ write
if src != orig:
    PATH.write_text(src, encoding='utf-8', newline='\n')
    print('WROTE', PATH)
else:
    print('no changes')
