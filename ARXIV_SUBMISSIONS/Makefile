# ARXIV_SUBMISSIONS/Makefile — drives self-contained per-paper builds.
#
# Each paper-X subfolder is independent. From the master folder:
#
#   make           # build all four PDFs
#   make A         # build only paper-A
#   make submit-A  # produce arxiv-ready tarball at dist/paper-A.tar.gz
#   make clean     # remove latex aux files
#   make distclean # also remove PDFs and dist/

PAPERS := A B C D
PDFS   := $(addprefix paper-,$(PAPERS:%=%/main.pdf))

LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -bibtex -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean A B C D $(addprefix submit-,$(PAPERS))

all: $(PDFS)

A: paper-A/main.pdf
B: paper-B/main.pdf
C: paper-C/main.pdf
D: paper-D/main.pdf

paper-%/main.pdf: paper-%/main.tex paper-%/hypertensor.sty paper-%/refs.bib
	cd $(dir $<) && $(LATEXMK) $(LATEXMK_FLAGS) main.tex

dist:
	mkdir -p dist

# Submit target: produce a flattened tarball with main.tex (comments
# stripped), main.bbl, and the .sty. arXiv does not run biber on uploads,
# so .bbl must be shipped.
submit-%: paper-%/main.pdf dist
	@echo "Preparing arXiv tarball for paper-$*..."
	@cd paper-$* && bash ../strip_comments.sh main.tex > main.cleaned.tex
	@cd paper-$* && mv main.tex main.tex.orig && mv main.cleaned.tex main.tex
	@cd paper-$* && $(LATEXMK) $(LATEXMK_FLAGS) main.tex
	@tar czf dist/paper-$*.tar.gz \
	    -C paper-$* main.tex main.bbl hypertensor.sty
	@cd paper-$* && mv main.tex.orig main.tex
	@cd paper-$* && $(LATEXMK) -c >/dev/null 2>&1 || true
	@echo "Built dist/paper-$*.tar.gz"

clean:
	@for d in $(addprefix paper-,$(PAPERS)); do \
	    (cd $$d && $(LATEXMK) -c >/dev/null 2>&1 || true); \
	done

distclean: clean
	@rm -f $(PDFS)
	@rm -rf dist
