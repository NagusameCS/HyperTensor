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

#!/usr/bin/env python3
"""
Merge all ARXIV_SUBMISSIONS paper .tex files into a single cohesive volume.

Strips individual preambles, extracts body content, and assembles
into volume_extended.tex with unified preamble.
"""
import os, re, glob, sys

BASE = r'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS'
OUTPUT = os.path.join(BASE, 'volume_extended.tex')

# Paper ordering: jury_proof first (foundation), then Papers I-XVIII
PAPER_ORDER = [
    # Foundation
    ('jury_proof.tex', 'A Mathematical Foundation for the Geometric Jury'),
    # Empirical Kernel (I-VI)
    ('paper-I/grc-attention-compression.tex', 'Paper I: GRC Attention Compression'),
    ('paper-II/geodesic-projection-pipeline.tex', 'Paper II: Geodesic Projection Pipeline'),
    ('paper-III/geodesic-speculative-decoding.tex', 'Paper III: Geodesic Speculative Decoding'),
    ('paper-IV/ott-gtc-manifold-runtime.tex', 'Paper IV: Organic Training Theory'),
    ('paper-V/grc-light-distillation.tex', 'Paper V: GRC Light Distillation'),
    ('paper-VI/task_level_impact.tex', 'Paper VI: Task-Level Impact'),
    # Extensions (VII-X)
    ('paper-VII/ffn_cluster_compression.tex', 'Paper VII: FFN Cluster Compression'),
    ('paper-VIII/gtc_as_rag.tex', 'Paper VIII: GTC as RAG'),
    ('paper-IX/super_baseline_general.tex', 'Paper IX: Cross-GPU Transfer'),
    ('paper-X/cmvb_splicing.tex', 'Paper X: CECI Model Grafting'),
    # k-Manifold Stack (XI-XV)
    ('paper-XI/ugt_taxonomy.tex', 'Paper XI: Universal Geodesic Taxonomy'),
    ('paper-XII/geodesic_compiler.tex', 'Paper XII: Native Geodesic Training'),
    ('paper-XIII/geodesic_synthesis.tex', 'Paper XIII: Safe OGD'),
    ('paper-XIV/geodesic_sniping.tex', 'Paper XIV: Behavioral Geodesic Sniping'),
    ('paper-XV/organic_generation.tex', 'Paper XV: COG + TEH'),
    # Riemann (XVI-XVIII)
    ('paper-XVI/agt_topology.tex', 'Paper XVI: AGT Topology of Zeta Zeros'),
    ('paper-XVII/acm_manifold.tex', 'Paper XVII: Analytic Continuation Manifold'),
    ('paper-XVIII/bridge_protocol.tex', 'Paper XVIII: The Bridge Protocol'),
]


def extract_body(filepath: str) -> str:
    """Extract body content between \\begin{document} and \\end{document}.
    Strips \\maketitle, \\title, \\author, \\date commands.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find document body
    begin_match = re.search(r'\\begin\{document\}', content)
    end_match = re.search(r'\\end\{document\}', content)
    
    if not begin_match:
        return f"% WARNING: No \\begin{{document}} found in {filepath}\n"
    
    body = content[begin_match.end():end_match.start() if end_match else None]
    
    # Strip \maketitle
    body = re.sub(r'\\maketitle\s*', '', body)
    
    # Strip \tableofcontents
    body = re.sub(r'\\tableofcontents\s*', '', body)
    
    # Strip \newpage commands that would break flow
    body = re.sub(r'\\newpage\s*', '\n', body)
    
    # Strip \input commands referencing external data files (fix paths)
    body = re.sub(r'\\input\{\.\./\.\./', r'\\input{../', body)
    
    # Strip the "macro to strip formatting" blocks
    body = re.sub(r'%\s*Macro to strip formatting.*?\n(?:\s*%.*\n)*', '', body)
    body = re.sub(r'\\makeatletter.*?\\makeatother\s*', '', body, flags=re.DOTALL)
    
    # Strip \appendix (would switch to Alph numbering and cause overflow)
    body = re.sub(r'\\appendix\s*', '', body)
    
    # Strip bold/italic formatting commands
    for cmd in ['textbf', 'textit', 'emph', 'mathbf', 'mathit', 'mathtt', 'mathsf', 'textsc', 'textup', 'textsl', 'bm']:
        body = re.sub(r'\\' + cmd + r'\{([^}]*)\}', r'\1', body)
    
    # Fix: \noindent directly followed by a letter (after \textit{} strip)
    body = re.sub(r'\\noindent(?=[A-Za-z])', r'\\noindent ', body)
    
    # Replace Unicode em/en dashes with LaTeX triple-dash (proper em dash)
    body = body.replace('\u2014', '---')
    body = body.replace('\u2013', '--')
    
    # Clean up excessive blank lines
    body = re.sub(r'\n{4,}', '\n\n\n', body)
    
    return body.strip()


def build_volume():
    """Build the merged volume."""
    lines = []
    
    #  Preamble  Use a known-good preamble (hypertensor.sty handles geometry)
    lines.append(r'\documentclass[11pt]{article}')
    lines.append(r'\usepackage[dvipsnames]{xcolor}')  # Colors for TikZ
    lines.append(r'\usepackage{amsmath,amssymb,graphicx,booktabs,enumitem,setspace}')
    lines.append(r'\usepackage{tikz}')  # For native figures
    lines.append(r'\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}')
    lines.append(r'\sloppy')  # Prevent margin overflows from unbreakable math/content
    lines.append(r'\setlength{\emergencystretch}{2em}')  # Extra stretch for overfull lines
    lines.append(r'\usepackage{hypertensor}')
    lines.append(r'\usepackage{lmodern}')  # Latin Modern — clean math, overrides hypertensor.sty newtx
    # Tighten margins after hypertensor loads geometry at 1.05in
    lines.append(r'\usepackage{geometry}')
    lines.append(r'\geometry{margin=0.75in}')
    # Matrix macros used across papers — provide if not already defined
    lines.append(r'\providecommand{\Amat}{\mathbf{A}}')
    lines.append(r'\providecommand{\Bmat}{\mathbf{B}}')
    lines.append(r'\providecommand{\Cmat}{\mathbf{C}}')
    lines.append(r'\providecommand{\Dmat}{\mathbf{D}}')
    lines.append(r'\providecommand{\Emat}{\mathbf{E}}')
    lines.append(r'\providecommand{\Fmat}{\mathbf{F}}')
    lines.append(r'\providecommand{\Gmat}{\mathbf{G}}')
    lines.append(r'\providecommand{\Hmat}{\mathbf{H}}')
    lines.append(r'\providecommand{\Imat}{\mathbf{I}}')
    lines.append(r'\providecommand{\Jmat}{\mathbf{J}}')
    lines.append(r'\providecommand{\Kmat}{\mathbf{K}}')
    lines.append(r'\providecommand{\Lmat}{\mathbf{L}}')
    lines.append(r'\providecommand{\Mmat}{\mathbf{M}}')
    lines.append(r'\providecommand{\Nmat}{\mathbf{N}}')
    lines.append(r'\providecommand{\Omat}{\mathbf{O}}')
    lines.append(r'\providecommand{\Pmat}{\mathbf{P}}')
    lines.append(r'\providecommand{\Qmat}{\mathbf{Q}}')
    lines.append(r'\providecommand{\Rmat}{\mathbf{R}}')
    lines.append(r'\providecommand{\Smat}{\mathbf{S}}')
    lines.append(r'\providecommand{\Tmat}{\mathbf{T}}')
    lines.append(r'\providecommand{\Umat}{\mathbf{U}}')
    lines.append(r'\providecommand{\Vmat}{\mathbf{V}}')
    lines.append(r'\providecommand{\Wmat}{\mathbf{W}}')
    lines.append(r'\providecommand{\Xmat}{\mathbf{X}}')
    lines.append(r'\providecommand{\Ymat}{\mathbf{Y}}')
    lines.append(r'\providecommand{\Zmat}{\mathbf{Z}}')
    lines.append(r'\providecommand{\Sigmamat}{\mathbf{\Sigma}}')
    lines.append(r'\providecommand{\Lambdamat}{\mathbf{\Lambda}}')
    lines.append(r'\providecommand{\Gammamat}{\mathbf{\Gamma}}')
    lines.append(r'\providecommand{\Deltamat}{\mathbf{\Delta}}')
    lines.append(r'\providecommand{\Thetamat}{\mathbf{\Theta}}')
    lines.append(r'\providecommand{\Phimat}{\mathbf{\Phi}}')
    lines.append(r'\providecommand{\Psimat}{\mathbf{\Psi}}')
    lines.append(r'\providecommand{\Omegamat}{\mathbf{\Omega}}')
    lines.append(r'\onehalfspacing')
    lines.append(r'\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}')
    lines.append(r'\pagestyle{empty}')
    lines.append(r'\renewcommand{\thesection}{\arabic{section}}')  # Avoid Alph overflow with 20+ papers
    lines.append(r'\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}')
    lines.append(r'\graphicspath{{figures/}}')  # Find figures relative to ARXIV_SUBMISSIONS
    
    # Biblatex compatibility: source papers may use natbib commands
    lines.append(r'\providecommand{\citep}[1]{\parencite{#1}}')
    lines.append(r'\providecommand{\citet}[1]{\textcite{#1}}')
    
    # Check for refs.bib — add to preamble if it has entries (biblatex requires preamble)
    refs_bib = os.path.join(BASE, 'refs.bib')
    if os.path.exists(refs_bib):
        with open(refs_bib, 'r', encoding='utf-8') as f:
            bib_content = f.read().strip()
        if bib_content and len(bib_content) > 10:  # Has actual entries
            lines.append(r'\bibliography{refs}')
    
    lines.append('')
    lines.append('% ')
    lines.append('% MERGED VOLUME -- All papers physically inserted')
    lines.append('% Generated by scripts/merge_volume.py')
    lines.append('% ')
    lines.append('')
    lines.append(r'\begin{document}')
    lines.append('')
    
    #  Title page 
    lines.append('\\title{HyperTensor: The Extended Volume\\\\\\large Papers I--XVIII with Mathematical Foundation}')
    lines.append('\\author{William Ken Ohara Stewart\\\\NagusameCS Independent Research\\\\\\texttt{https://github.com/NagusameCS/HyperTensor}}')
    lines.append('\\date{May 6, 2026}')
    lines.append('\\maketitle')
    lines.append('')
    
    #  Foreword (use section* to avoid numbering)
    lines.append(r'\section*{Foreword}')
    lines.append('')
    lines.append(r'Thank you for taking the time to look through this work. My name is William Ken Ohara Stewart. I am eighteen years old and a high school student in Mexico City. I am not a professional researcher, I do not have a lab or an advisor, and I am certain there are mistakes in these pages that a trained eye would catch immediately. I ask for your patience with those.')
    lines.append('')
    lines.append(r'Everything here was done on a laptop with an RTX 4070, a rented cloud GPU when a bigger one was needed, and a lot of headaches.')
    lines.append('')
    lines.append("This volume contains nineteen papers: eighteen research papers and a mathematical foundation for the geometric jury. Papers I through VI form the empirical kernel, VII through X extend the framework, XI through XV form the k-manifold living-model stack, and XVI through XVIII apply the geometric jury principle to the Riemann Hypothesis. Papers XVI--XVIII are presented as a geometric visualisation of the functional equation's $Z_2$ symmetry, not as a contribution to analytic number theory; see the explicit disclaimers in those papers' abstracts.")
    lines.append('')
    lines.append(r'\subsection*{Hardware}')
    lines.append(r'Measurements span three GPU classes: RTX 4070 Laptop (8GB VRAM, 32MB L2; AD106 spec), EC2 L40S (48GB VRAM, 48MB L2), and NVIDIA A10G (24GB). The RTX 4070 Laptop has a 32\,MB L2 cache (NVIDIA AD106 datasheet). On this GPU we observe an empirical optimum at $k^* = 1536$; the per-MB constant is therefore $1536/32 = 48.0$. The relation $k^* \approx \mathrm{L2\_MB} \times 48.0$ is consistent with measurements on EC2 L40S and A10G. We mark the constant as tentative pending cross-vendor measurement (AMD, Apple Silicon, TPU).')
    lines.append('')
    
    # Verification Scope subsection
    lines.append(r'\subsection*{Verification Scope}')
    lines.append(r'Quantitative claims in this volume are catalogued in the repository file \texttt{VERIFICATION\_STATUS.md} with one of four tags: REAL (measured on hardware), SIM (synthetic or Monte Carlo), MIXED (combination), or UNVERIFIED (compute-bound or outstanding). Claims tagged SIM or UNVERIFIED in that file are noted in the body where space permits. Verification infrastructure is in \texttt{scripts/comprehensive\_verify.py}. See also \texttt{BENCHMARK\_PROTOCOL.md} and \texttt{COMPREHENSIVE\_STATE.md} in the repository root.')
    lines.append('')
    
    # Paper listing instead of TOC (TOC creates blank pages with 217 sections)
    lines.append(r'\subsection*{Papers in This Volume}')
    lines.append('')
    for i, (rel_path, section_title) in enumerate(PAPER_ORDER, 1):
        short = section_title.replace('Paper ','').replace('Jury Proof: ','').replace('Appendix: ','')
        lines.append(f'{i}. {short} \\\\')
    lines.append('')
    
    #  Glossary 
    glossary_path = os.path.join(BASE, '..', 'docs', 'explained_terms.txt')
    if os.path.exists(glossary_path):
        lines.append(r'\section*{Glossary}')
        lines.append('')
        with open(glossary_path, 'r', encoding='utf-8', errors='replace') as gf:
            glossary_text = gf.read()
        # Sanitize Unicode characters that LaTeX can't handle directly
        unicode_to_latex = {
            '\u2081': '$_{1}$', '\u2082': '$_{2}$', '\u2083': '$_{3}$',
            '\u03B6': '$\\zeta$', '\u03B9': '$\\iota$', '\u03C7': '$\\chi$',
            '\u03A6': '$\\Phi$', '\u03BB': '$\\lambda$',
            '\u2212': '$-$', '\u221E': '$\\infty$',
            '\u03C3': '$\\sigma$',
            '\u00B2': '$^{2}$', '\u00B7': '$\\cdot$',
            '\u00D7': '$\\times$', '\u00B0': '$^{\\circ}$',
        }
        for uni, latex in unicode_to_latex.items():
            glossary_text = glossary_text.replace(uni, latex)
        # Escape underscores in text mode (they become subscripts in LaTeX)
        # Replace _ with \\_ unless it's already inside $...$
        # Simple approach: split on $, escape _ in even segments (text mode)
        def escape_text_specials(text):
            segments = text.split('$')
            for i in range(0, len(segments), 2):  # Even indices are text mode
                segments[i] = segments[i].replace('_', '\\_')
                segments[i] = segments[i].replace('^', '\\^{}')
                segments[i] = segments[i].replace('#', '\\#')
                segments[i] = segments[i].replace('&', '\\&')
            return '$'.join(segments)
        glossary_text = escape_text_specials(glossary_text)
        # Parse glossary: lines with "term: definition" format
        current_category = ''
        for line in glossary_text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') and not stripped.startswith('##'):
                # Section header like "#  Jury Proof "
                current_category = stripped.replace('#', '').strip()
                if current_category and not current_category.startswith('='):
                    lines.append(r'\subsection*{' + current_category + '}')
                    lines.append('')
            elif ':' in stripped and not stripped.startswith('#'):
                parts = stripped.split(':', 1)
                term = parts[0].strip()
                definition = parts[1].strip()
                if term and definition:
                    lines.append(r'\textbf{' + term + '}: ' + definition + r'\\[2pt]')
        lines.append('')
    lines.append('\\newpage')
    lines.append('')
    lines.append('\\newpage')
    lines.append('')
    
    #  Insert each paper 
    paper_num = 0
    for rel_path, section_title in PAPER_ORDER:
        paper_num += 1
        filepath = os.path.join(BASE, rel_path)
        
        if not os.path.exists(filepath):
            lines.append(f'% WARNING: File not found: {rel_path}')
            continue
        
        print(f'  [{paper_num:2d}/{len(PAPER_ORDER)}] {rel_path}...')
        
        body = extract_body(filepath)
        
        lines.append('')
        lines.append('%' + '-' * 65)
        lines.append(f'% {section_title}')
        lines.append('%' + '-' * 65)
        lines.append('')
        lines.append('\\newpage')
        
        # Determine section level with clear visual separation
        if 'jury_proof' in rel_path or 'mathematician_handoff' in rel_path:
            lines.append(f'\\section*{{{section_title}}}')
        else:
            lines.append(f'\\section{{{section_title}}}')
        lines.append('\\vspace{6pt}')
        lines.append('\\hrule')
        lines.append('\\vspace{12pt}')
        lines.append('')
        
        # Insert body
        lines.append(body)
        lines.append('')
        lines.append('\\newpage')
        lines.append('')
    
    #  Unified bibliography — use printbibliography (biblatex, loaded by hypertensor.sty)
    lines.append('')
    lines.append('% ')
    lines.append('% BIBLIOGRAPHY')
    lines.append('% ')
    lines.append('')
    
    # Check for refs.bib — only include if it actually has entries
    refs_bib = os.path.join(BASE, 'refs.bib')
    if os.path.exists(refs_bib):
        with open(refs_bib, 'r', encoding='utf-8') as f:
            bib_content = f.read().strip()
        if bib_content and len(bib_content) > 10:  # Has actual entries
            lines.append(r'\printbibliography')
        else:
            lines.append('% Bibliography file empty — omitted')
    else:
        lines.append('% refs.bib not found — bibliography omitted')
    
    #  Back matter: Negative Results, Data Availability, COI, Verification Status 
    lines.append('')
    lines.append(r'\section*{Appendix: Negative Results and Falsified Predictions}')
    lines.append(r'In the spirit of self-falsification urged by the peer review of this volume, we record here predictions and prototypes that did not survive empirical contact. Each entry names the prediction, the experiment, the outcome, and the corresponding implication for the surrounding paper.')
    lines.append('')
    lines.append(r'\paragraph{N1. Curvature-warp local Christoffel injection (Paper IV).} \textbf{Prediction:} that a local Gaussian-decayed metric warp around a knowledge-injection point would redirect the geodesic so that the injected fact survives a next-token decode without measurable spillover onto unrelated decode targets. \textbf{Experiment:} 32-configuration sweep over (warp strength, $\sigma$, characteristic length $d_\ell$) on SmolLM2-135M; success criterion was decode-aligned MRR improvement $\geq$ a fixed threshold without spillover divergence. \textbf{Outcome:} 0/32 configurations passed. The best per-injection improvement was 16\%; spillover diverged at high warp strength. \textbf{Mechanism:} the SmolLM2 manifold is sufficiently flat at the relevant scale that a local Gaussian metric warp cannot redirect the geodesic without measurable global side-effects on other decode trajectories. \textbf{Implication for Paper IV:} the local-warp variant of OTT is not a viable knowledge-injection mechanism on this model. The accumulated-warp persistence machinery (\texttt{axiomwarpstate.dat}) and the GTC cache that Paper IV builds on remain unaffected. Source: \texttt{docs/figures/curvaturewarp/}, \texttt{scripts/curvaturewarp/sweep.py}.')
    lines.append('')
    lines.append(r'\paragraph{N2. Within-model CECI grafting on SmolLM2 (Paper X).} \textbf{Prediction:} that a within-model CECI graft (sub-network swapped between two checkpoints of the same architecture) would preserve perplexity within tolerance. \textbf{Experiment:} 120 within-model graft attempts on SmolLM2-135M. \textbf{Outcome:} 0/120 grafts passed. \textbf{Implication for Paper X:} CECI viability appears to require cross-architecture graft conditions; the headline cross-graft results in Paper X should be read as conditional on those particular architecture pairs and not as a general claim about within-architecture grafting. The negative within-model series is the load-bearing falsification for the scope statement in Paper~X.')
    lines.append('')
    lines.append(r'\paragraph{N3. L2-cache-fit as the dominant cause of the GRC $k^\star$ optimum (Paper I).} \textbf{Prediction:} that the empirically observed $k^\star = 1536$ optimum on RTX 4070 Laptop (32\,MB L2) was \emph{primarily} explained by working-set L2-residency. \textbf{Experiment:} ablation in which kernel fusion was disabled while $k$ was held at the cache-fit optimum. \textbf{Outcome:} the dominant share of the speedup is attributable to kernel fusion, not L2 residency; cache-fit is a secondary effect. The Paper I body (\S kernel-fusion analysis) attributes the dominant share to fusion. \textbf{Implication:} the cross-vendor scaling formula $k^\star \approx \mathrm{L2\_MB} \times 48.0$ is presented as \emph{tentative pending cross-vendor measurement}; AMD/Apple-Silicon/TPU runs may break the linear fit. The Paper I abstract still mentions L2 cache before kernel fusion --- this is a known residual phrasing inconsistency, flagged in the gap inventory.')
    lines.append('')
    lines.append(r"\paragraph{N4. UGT random-$B'$ ablation (Papers XII--XV) --- confirmed null.} \textbf{Prediction:} that substituting a uniformly-random orthonormal basis $B'$ for the UGT-derived basis $B$ in the four cascading papers would destroy the matched-diagonal zone-ablation signal, demonstrating that the UGT basis is semantically meaningful (zone~1 supports syntax probes, zone~2 algorithmic, zone~3 factual) rather than merely a low-rank parameter-efficient subspace. \textbf{Experiment:} \texttt{scripts/ugt\_random\_basis\_ablation.py}. Per-probe paired contrast $\Delta = \mathrm{logit\,delta}_B - \mathrm{logit\,delta}_{B'}$ for each (category, ablated zone) cell, paired-$t$ and Wilcoxon tests over (seed, probe). Four runs: (i)~SmolLM2-135M, $n{=}8$ seeds, default 9-probe suite, 24 paired diffs/cell; (ii)~SmolLM2-135M, $n{=}8$ seeds, extended 30-probe suite, 24 paired diffs/cell; (iii)~Qwen2.5-0.5B-Instruct, $n{=}5$ seeds, extended suite, 50 paired diffs/cell; (iv)~SmolLM2-135M, $n{=}5$ seeds, $1500$ training steps with $\lambda_{\mathrm{TOP}}=0.5$ (vs.\ default $0.05$), extended suite, 50 paired diffs/cell. \textbf{Outcome:} no diagonal cell reaches significance in any run. Smallest diagonal $p$-values (paired $t$): SmolLM2 short~$0.143$ (factual/factual, $\Delta{=}{+}0.020$); Qwen $0.270$ (factual/factual, $\Delta{=}{-}0.009$); SmolLM2 long-strong-TOP $0.257$ (factual/factual, $\Delta{=}{+}0.004$). All cells fail the Bonferroni-adjusted threshold $\alpha{=}0.0056$ for FWER$=0.05$ over 9 cells. The lone nominally-significant cell across all runs (algorithmic-category / syntax-zone, $p{=}0.020$) is off-diagonal, negatively signed, and small, consistent with multiple-comparisons noise. \textbf{Diagnostic:} the TOP ``purity'' metric returns $1.0$ for the random orthonormal $B'$ as well, because purity as currently defined measures column orthogonality and a Haar-random orthonormal frame is by construction orthonormal. Strengthening TOP regularisation by an order of magnitude and tripling training steps (run~iv) does not produce the predicted diagonal signal. \textbf{Stronger layer-wise intervention (run~v):} to address the residual concern that runs~i--iv only intervene at the final hidden state and therefore touch at most ${\sim}k/d \approx 5\%$ of the residual stream, we re-ran the ablation with a forward hook on \emph{every} transformer block subtracting the zone projection at the residual stream at every layer (\texttt{scripts/ugt\_random\_basis\_layerwise.py}), which is the standard mechanistic-interpretability intervention used by RepE, abliteration, ROME, and MEMIT. SmolLM2-135M, $n{=}5$ seeds, extended 30-probe suite, 50 paired diffs/cell. Per-cell effect sizes are ${\sim}10\times$ larger than runs~i--iv, confirming the intervention is biting. Diagonal cells: syntax/syntax $\Delta{=}{-}0.0094$ ($t_p{=}0.172$, Wilcoxon $p{=}0.013$); algorithmic/algorithmic $\Delta{=}{-}0.0080$ ($t_p{=}0.088$, Wilcoxon $p{=}0.009$); factual/factual $\Delta{=}{+}0.0047$ ($t_p{=}0.511$, Wilcoxon $p{=}0.781$). Two of three diagonals reach nominal Wilcoxon significance \emph{but with the opposite sign} from the H_meaningful prediction (B hurts \emph{less} than random B' on its predicted-zone diagonal); neither survives Bonferroni $\alpha{=}0.0056$. The likely mechanism is that TOP regularisation drives B toward low-variance / low-impact directions of the residual stream so as to keep the LM cross-entropy down during basis training, making B-projections \emph{less} damaging than Haar-random projections rather than more. \textbf{Scaled layer-wise replication at 7B (run~vi):} to address the residual concern that 135M is too small for the meaningful-basis claim to manifest, we replicated the layer-wise intervention on Qwen2.5-7B-Instruct (4-bit NF4 quantisation to fit 8\,GB VRAM) with $k$ scaled to preserve the residual fraction $k/d \approx 5.6\%$ ($k{=}200$, zones $\{66,133,200\}$, $n{=}3$ seeds, extended 30-probe suite, 90 paired diffs/cell). Diagonal cells: syntax/syntax $\Delta{=}{-}0.1364$ ($t_p{=}0.038$, Wilcoxon $p{=}0.004$, CI95 $[-0.262,-0.021]$); algorithmic/algorithmic $\Delta{\approx}{-}10^{-7}$; factual/factual $\Delta{=}{+}0.024$ ($t_p{=}0.217$). The syntax/syntax cell at 7B \emph{crosses} the Bonferroni-adjusted threshold $\alpha{=}0.0056$ \emph{but with the opposite sign from H_meaningful}: at 7B scale the UGT basis B is, on its own predicted-zone probes, $13.6$ percentage points \emph{less} damaging than a Haar-random orthonormal basis when ablated layer-wise. This is a Bonferroni-significant falsification of the meaningful-basis claim at production-scale model size, with the failure direction (B hurts less, not more) consistent across SmolLM2-135M and Qwen2.5-7B and consistent with the diagnostic mechanism above. \textbf{Reframe analysis (run~vii):} we additionally tested three weakened forms of the meaningful-basis claim on the layer-wise SmolLM2 data (\texttt{scripts/ugt\_reframe\_analysis.py}). R1 (B-as-privileged-subspace, ignoring zone labels): pooled paired diff over all 9 cells mean${=}{-}0.0011$, $t_p{=}0.64$ — not supported. R2 (zones-differ-but-mislabeled): one-way ANOVA over the three zones under B yields $F{=}0.012$, $p{=}0.988$, with the three zones producing statistically indistinguishable mean damage (means $0.0066$, $0.0068$, $0.0061$); zones are not even three different things from each other. R3 (best relabeling): the maximum mean diagonal $(B{-}B')$ achievable by any 1-1 permutation of the three zone labels is $+0.0041$, smaller than the per-cell noise floor and not the named identity permutation. No relabeling rescues the claim. \textbf{Implication for Papers XII--XV:} across six runs spanning two model families (Llama and Qwen), three model scales (135M, 0.5B, 7B), two ablation procedures (final-state and layer-wise), two probe batteries (9 and 30 probes), two basis-training intensities (default and 10$\times$ stronger TOP), and three weakened reframings, the UGT-as-meaningful-basis claim is not supported, and at 7B scale is Bonferroni-significantly falsified in the opposite direction. The four cascading papers should be read as low-rank parameter-efficiency results in which the basis acts as a generic compressed subspace, rather than as evidence that the UGT-defined zones carry independent semantic structure. The original UGT-as-meaningful-basis claim is not supported by the measurements reported here. The narrower compression and parameter-efficiency results in Papers XII--XV are unaffected by this null. Result files: \texttt{benchmarks/ugt\_random\_basis\_ablation\_smol135m\_n8.json}, \texttt{ugt\_random\_basis\_ablation\_qwen500m\_ext\_n5.json}, \texttt{ugt\_random\_basis\_smol135m\_long\_strongtop\_n5.json}, \texttt{ugt\_random\_basis\_layerwise\_smol135m\_ext\_n5.json}, \texttt{ugt\_random\_basis\_layerwise\_qwen7b\_k200\_ext\_n3.json}, \texttt{ugt\_reframe\_analysis.json}.")
    lines.append('')
    lines.append(r"\paragraph{N5. Riemann-papers analytic claim.} The combined Riemann arc (Papers XVI--XVIII) does \emph{not} reduce the Riemann Hypothesis to an attainable next-step theorem. The ``necessity argument'' in Paper XVII is conditional on the faithfulness of the learned $h$, which is open. We make no analytic claim regarding RH. This is recorded as a negative result against any reading of the volume that infers a partial proof.")
    lines.append('')
    lines.append(r'\section*{Data Availability}')
    lines.append(r'All code, benchmarks, and reproduction packages are at \texttt{https://github.com/NagusameCS/HyperTensor}. Benchmarks are under \texttt{benchmarks/}; individual paper reproduction packages are versioned (e.g., \texttt{whitepaper\_pack\_20260427\_121815}). Models are from HuggingFace: \texttt{Qwen/Qwen2.5-0.5B-Instruct}, \texttt{Qwen/Qwen2.5-1.5B-Instruct}, \texttt{HuggingFaceTB/SmolLM2-135M}, \texttt{meta-llama/Llama-3.1-8B}. Hardware: RTX 4070 Laptop (8\,GB VRAM, 32\,MB L2; AD106), EC2 L40S (48\,GB), EC2 A10G (24\,GB).')
    lines.append('')
    lines.append(r'\section*{Funding and Competing Interests}')
    lines.append(r'Funding: none. Competing interests: none. Affiliations: independent researcher.')
    lines.append('')
    lines.append(r'\section*{Verification Status}')
    lines.append(r'Quantitative claims in this volume are catalogued in the repository file \texttt{VERIFICATION\_STATUS.md} with one of four tags: REAL (measured on hardware), SIM (synthetic or Monte Carlo), MIXED (combination), or UNVERIFIED (compute-bound or outstanding). Claims tagged SIM or UNVERIFIED in that file are noted as such in the body where space permits. The verification infrastructure is in \texttt{scripts/comprehensive\_verify.py}. See also \texttt{BENCHMARK\_PROTOCOL.md} and \texttt{COMPREHENSIVE\_STATE.md} in the repository root.')
    lines.append('')
    lines.append('\\end{document}')
    
    #  Write 
    output = '\n'.join(lines)
    
    # Backup original (only if it exists)  
    existing = os.path.join(BASE, 'volume_extended.tex')
    backup = existing + '.bak'
    if os.path.exists(existing) and not os.path.exists(backup):
        import shutil
        shutil.copy2(existing, backup)
        print(f'\n  Backed up original to volume_extended.tex.bak')
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f'\n  Written: {OUTPUT}')
    print(f'  Size: {len(output):,} bytes, {output.count(chr(10)):,} lines')
    print(f'  Papers merged: {paper_num}')


if __name__ == '__main__':
    build_volume()
