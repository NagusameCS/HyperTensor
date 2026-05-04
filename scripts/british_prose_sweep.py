"""Regex-based British English sweep on docs/*.html.

Skips <style>/<script>/<code>/<pre> blocks, $...$ math, \\(...\\), \\[...\\],
$$...$$, and HTML attribute values (i.e. anything inside any HTML tag).
Applies replacements only on the prose between tags.
"""
from __future__ import annotations
import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs"

PAIRS = [
    ("organize", "organise"), ("organized", "organised"),
    ("organizing", "organising"), ("organization", "organisation"),
    ("organizations", "organisations"),
    ("optimize", "optimise"), ("optimized", "optimised"),
    ("optimizing", "optimising"), ("optimization", "optimisation"),
    ("optimizations", "optimisations"),
    ("normalize", "normalise"), ("normalized", "normalised"),
    ("normalizing", "normalising"), ("normalization", "normalisation"),
    ("analyze", "analyse"), ("analyzed", "analysed"),
    ("analyzing", "analysing"),
    ("realize", "realise"), ("realized", "realised"),
    ("realizing", "realising"),
    ("recognize", "recognise"), ("recognized", "recognised"),
    ("recognizing", "recognising"),
    ("minimize", "minimise"), ("minimized", "minimised"),
    ("minimizing", "minimising"),
    ("maximize", "maximise"), ("maximized", "maximised"),
    ("maximizing", "maximising"),
    ("parameterize", "parameterise"), ("parameterized", "parameterised"),
    ("parameterizing", "parameterising"),
    ("characterize", "characterise"), ("characterized", "characterised"),
    ("characterizing", "characterising"),
    ("utilize", "utilise"), ("utilized", "utilised"),
    ("utilizing", "utilising"),
    ("summarize", "summarise"), ("summarized", "summarised"),
    ("summarizing", "summarising"),
    ("modeling", "modelling"), ("modeled", "modelled"),
    ("labeling", "labelling"), ("labeled", "labelled"),
    ("behavior", "behaviour"), ("behaviors", "behaviours"),
    ("behavioral", "behavioural"),
    ("flavor", "flavour"), ("flavors", "flavours"),
    ("favor", "favour"), ("favors", "favours"),
    ("favored", "favoured"), ("favoring", "favouring"),
    ("neighbor", "neighbour"), ("neighbors", "neighbours"),
    ("neighboring", "neighbouring"),
    ("specialize", "specialise"), ("specialized", "specialised"),
    ("specialization", "specialisation"),
    ("generalize", "generalise"), ("generalized", "generalised"),
    ("generalization", "generalisation"),
    ("factorize", "factorise"), ("factorized", "factorised"),
    ("factorization", "factorisation"),
    ("modernize", "modernise"), ("modernized", "modernised"),
    ("serialize", "serialise"), ("serialized", "serialised"),
    ("serialization", "serialisation"),
    ("tokenize", "tokenise"), ("tokenized", "tokenised"),
    ("tokenization", "tokenisation"),
    ("visualize", "visualise"), ("visualized", "visualised"),
    ("visualization", "visualisation"),
    ("discretize", "discretise"), ("discretized", "discretised"),
    ("discretization", "discretisation"),
    ("regularize", "regularise"), ("regularized", "regularised"),
    ("regularization", "regularisation"),
    ("memorize", "memorise"), ("memorized", "memorised"),
    ("standardize", "standardise"), ("standardized", "standardised"),
    ("standardization", "standardisation"),
    ("initialize", "initialise"), ("initialized", "initialised"),
    ("initialization", "initialisation"),
    ("synchronize", "synchronise"), ("synchronized", "synchronised"),
    ("synchronization", "synchronisation"),
    ("authorize", "authorise"), ("authorized", "authorised"),
    ("authorization", "authorisation"),
    ("emphasize", "emphasise"), ("emphasized", "emphasised"),
    ("emphasizing", "emphasising"),
    ("prioritize", "prioritise"), ("prioritized", "prioritised"),
    ("prioritizing", "prioritising"),
    ("hypothesize", "hypothesise"), ("hypothesized", "hypothesised"),
    ("quantization", "quantisation"),
    ("judgment", "judgement"),
    ("defense", "defence"), ("offense", "offence"),
    ("aluminum", "aluminium"),
    ("inquire", "enquire"), ("inquiry", "enquiry"),
    ("artifact", "artefact"), ("artifacts", "artefacts"),
]


def _build():
    out = []
    for a, b in PAIRS:
        out.append((re.compile(r"\b" + re.escape(a) + r"\b"), b))
        out.append((re.compile(r"\b" + re.escape(a.title()) + r"\b"), b.title()))
    return out


REPLS = _build()


SKIP_RE = re.compile(
    r"(<style\b[^>]*>.*?</style>"
    r"|<script\b[^>]*>.*?</script>"
    r"|<code\b[^>]*>.*?</code>"
    r"|<pre\b[^>]*>.*?</pre>"
    r"|]*>.*?"
    r"|<section\b[^>]*\bid=\"references\"[^>]*>.*?</section>"
    r"|<section\b[^>]*\bid=\"refs\"[^>]*>.*?</section>"
    r"|<div\b[^>]*\bid=\"refs\"[^>]*>(?:[^<]|<(?!/?div\b))*?</div>\s*</section>"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]*\$"
    r"|\\\(.*?\\\)"
    r"|\\\[.*?\\\]"
    r"|<[^>]*>)",
    re.DOTALL | re.IGNORECASE,
)


def transform_text(text: str) -> str:
    for rx, b in REPLS:
        text = rx.sub(b, text)
    return text


def process_html(src: str) -> str:
    out = []
    pos = 0
    for m in SKIP_RE.finditer(src):
        out.append(transform_text(src[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(transform_text(src[pos:]))
    return "".join(out)


def main():
    targets = []
    for sub in ("papers", "research"):
        targets += list((ROOT / sub).glob("*.html"))
    targets += [ROOT / "index.html", ROOT / "whitepaper.html"]
    changed = 0
    for p in targets:
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        new = process_html(src)
        if new != src:
            p.write_text(new, encoding="utf-8")
            changed += 1
            print(f"updated {p.relative_to(ROOT)}")
    print(f"\nchanged {changed} of {len(targets)} files")


if __name__ == "__main__":
    main()
