# Master arXiv submission Makefile — HyperTensor Parts I-XV
LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode

PAPERS := I II III IV V VI VII VIII IX X XI XII XIII XIV XV
TEX_I    := grc-attention-compression
TEX_II   := geodesic-projection-pipeline
TEX_III  := geodesic-speculative-decoding
TEX_IV   := ott-gtc-manifold-runtime
TEX_V    := grc-light-distillation
TEX_VI   := task_level_impact
TEX_VII  := ffn_cluster_compression
TEX_VIII := gtc_as_rag
TEX_IX   := super_baseline_general
TEX_X    := cmvb_splicing
TEX_XI   := ugt_taxonomy
TEX_XII  := geodesic_compiler
TEX_XIII := geodesic_synthesis
TEX_XIV  := geodesic_sniping
TEX_XV   := organic_generation

.PHONY: all clean $(PAPERS)
all: $(PAPERS)

I: paper-I/$(TEX_I).pdf
II: paper-II/$(TEX_II).pdf
III: paper-III/$(TEX_III).pdf
IV: paper-IV/$(TEX_IV).pdf
V: paper-V/$(TEX_V).pdf
VI: paper-VI/$(TEX_VI).pdf
VII: paper-VII/$(TEX_VII).pdf
VIII: paper-VIII/$(TEX_VIII).pdf
IX: paper-IX/$(TEX_IX).pdf
X: paper-X/$(TEX_X).pdf
XI: paper-XI/$(TEX_XI).pdf
XII: paper-XII/$(TEX_XII).pdf
XIII: paper-XIII/$(TEX_XIII).pdf
XIV: paper-XIV/$(TEX_XIV).pdf
XV: paper-XV/$(TEX_XV).pdf

paper-I/$(TEX_I).pdf: paper-I/$(TEX_I).tex paper-I/hypertensor.sty paper-I/refs.bib
	cd paper-I && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_I).tex
paper-II/$(TEX_II).pdf: paper-II/$(TEX_II).tex paper-II/hypertensor.sty paper-II/refs.bib
	cd paper-II && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_II).tex
paper-III/$(TEX_III).pdf: paper-III/$(TEX_III).tex paper-III/hypertensor.sty paper-III/refs.bib
	cd paper-III && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_III).tex
paper-IV/$(TEX_IV).pdf: paper-IV/$(TEX_IV).tex paper-IV/hypertensor.sty paper-IV/refs.bib
	cd paper-IV && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_IV).tex
paper-V/$(TEX_V).pdf: paper-V/$(TEX_V).tex paper-V/hypertensor.sty paper-V/refs.bib
	cd paper-V && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_V).tex
paper-VI/$(TEX_VI).pdf: paper-VI/$(TEX_VI).tex paper-VI/hypertensor.sty paper-VI/refs.bib
	cd paper-VI && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_VI).tex
paper-VII/$(TEX_VII).pdf: paper-VII/$(TEX_VII).tex paper-VII/hypertensor.sty paper-VII/refs.bib
	cd paper-VII && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_VII).tex
paper-VIII/$(TEX_VIII).pdf: paper-VIII/$(TEX_VIII).tex paper-VIII/hypertensor.sty paper-VIII/refs.bib
	cd paper-VIII && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_VIII).tex
paper-IX/$(TEX_IX).pdf: paper-IX/$(TEX_IX).tex paper-IX/hypertensor.sty paper-IX/refs.bib
	cd paper-IX && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_IX).tex
paper-X/$(TEX_X).pdf: paper-X/$(TEX_X).tex paper-X/hypertensor.sty paper-X/refs.bib
	cd paper-X && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_X).tex
paper-XI/$(TEX_XI).pdf: paper-XI/$(TEX_XI).tex paper-XI/hypertensor.sty paper-XI/refs.bib
	cd paper-XI && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_XI).tex
paper-XII/$(TEX_XII).pdf: paper-XII/$(TEX_XII).tex paper-XII/hypertensor.sty paper-XII/refs.bib
	cd paper-XII && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_XII).tex
paper-XIII/$(TEX_XIII).pdf: paper-XIII/$(TEX_XIII).tex paper-XIII/hypertensor.sty paper-XIII/refs.bib
	cd paper-XIII && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_XIII).tex
paper-XIV/$(TEX_XIV).pdf: paper-XIV/$(TEX_XIV).tex paper-XIV/hypertensor.sty paper-XIV/refs.bib
	cd paper-XIV && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_XIV).tex
paper-XV/$(TEX_XV).pdf: paper-XV/$(TEX_XV).tex paper-XV/hypertensor.sty paper-XV/refs.bib
	cd paper-XV && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_XV).tex

clean:
	@for P in $(PAPERS); do cd paper-$$P && $(LATEXMK) -c 2>/dev/null || true; done
	@echo "Cleaned all aux files."

clean:
	@for d in paper-A paper-B paper-C paper-D; do (cd $$d && $(LATEXMK) -c >/dev/null 2>&1 || true); done

distclean: clean
	@rm -rf dist
	@for d in paper-A paper-B paper-C paper-D; do (cd $$d && $(LATEXMK) -C >/dev/null 2>&1 || true); done
