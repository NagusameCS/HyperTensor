# Master arXiv submission Makefile (self-contained per-paper folders).
LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

PAPERS := A B C D
TEX_A := grc-attention-compression
TEX_B := geodesic-projection-pipeline
TEX_C := geodesic-speculative-decoding
TEX_D := ott-gtc-manifold-runtime

.PHONY: all clean distclean A B C D $(addprefix submit-,$(PAPERS))

all: A B C D
A: paper-A/$(TEX_A).pdf
B: paper-B/$(TEX_B).pdf
C: paper-C/$(TEX_C).pdf
D: paper-D/$(TEX_D).pdf

paper-A/$(TEX_A).pdf: paper-A/$(TEX_A).tex paper-A/hypertensor.sty paper-A/refs.bib
	cd paper-A && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_A).tex
paper-B/$(TEX_B).pdf: paper-B/$(TEX_B).tex paper-B/hypertensor.sty paper-B/refs.bib
	cd paper-B && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_B).tex
paper-C/$(TEX_C).pdf: paper-C/$(TEX_C).tex paper-C/hypertensor.sty paper-C/refs.bib
	cd paper-C && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_C).tex
paper-D/$(TEX_D).pdf: paper-D/$(TEX_D).tex paper-D/hypertensor.sty paper-D/refs.bib
	cd paper-D && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX_D).tex

dist:
	mkdir -p dist

submit-A: A | dist
	@$(MAKE) _submit PAPER=A TEX=$(TEX_A)
submit-B: B | dist
	@$(MAKE) _submit PAPER=B TEX=$(TEX_B)
submit-C: C | dist
	@$(MAKE) _submit PAPER=C TEX=$(TEX_C)
submit-D: D | dist
	@$(MAKE) _submit PAPER=D TEX=$(TEX_D)

# Strip comments, rebuild for fresh .bbl, then tar source + bbl + sty + figures.
_submit:
	@cd paper-$(PAPER) && bash ../strip_comments.sh $(TEX).tex > $(TEX).cleaned.tex
	@cd paper-$(PAPER) && mv $(TEX).tex $(TEX).tex.orig && mv $(TEX).cleaned.tex $(TEX).tex
	@cd paper-$(PAPER) && $(LATEXMK) $(LATEXMK_FLAGS) $(TEX).tex
	@FIGS=""; if [ -d paper-$(PAPER)/figures ]; then FIGS="figures"; fi; \
	  tar czf dist/paper-$(PAPER).tar.gz -C paper-$(PAPER) $(TEX).tex $(TEX).bbl hypertensor.sty $$FIGS
	@cd paper-$(PAPER) && mv $(TEX).tex.orig $(TEX).tex
	@cd paper-$(PAPER) && $(LATEXMK) -c >/dev/null 2>&1 || true
	@echo "  -> dist/paper-$(PAPER).tar.gz"

clean:
	@for d in paper-A paper-B paper-C paper-D; do (cd $$d && $(LATEXMK) -c >/dev/null 2>&1 || true); done

distclean: clean
	@rm -rf dist
	@for d in paper-A paper-B paper-C paper-D; do (cd $$d && $(LATEXMK) -C >/dev/null 2>&1 || true); done
