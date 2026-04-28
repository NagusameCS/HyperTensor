# HyperTensor arXiv build
#
# Usage:
#   make             # build all four PDFs
#   make A B C D     # named papers
#   make clean       # remove latexmk auxiliaries
#   make distclean   # also remove PDFs
#   make submit-A    # build flattened arXiv tarball for paper A

LATEXMK := latexmk
LMKFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

PAPERS := A B C D

DIR_A := paperA-attention-compression
DIR_B := paperB-geodesic-projection
DIR_C := paperC-speculative-decoding
DIR_D := paperD-ott-gtc

ALL_DIRS := $(DIR_A) $(DIR_B) $(DIR_C) $(DIR_D)

.PHONY: all clean distclean A B C D $(addprefix submit-,$(PAPERS))

all: A B C D

A: $(DIR_A)/main.pdf
B: $(DIR_B)/main.pdf
C: $(DIR_C)/main.pdf
D: $(DIR_D)/main.pdf

# Generic rule -- build inside each paper directory so relative \input paths work.
%/main.pdf: %/main.tex common/hypertensor.sty common/refs.bib
	cd $(dir $<) && $(LATEXMK) $(LMKFLAGS) main.tex

clean:
	@for d in $(ALL_DIRS); do \
	  cd $$d && $(LATEXMK) -c >/dev/null 2>&1 ; cd .. ; \
	done
	@rm -rf dist

distclean: clean
	@for d in $(ALL_DIRS); do \
	  cd $$d && $(LATEXMK) -C >/dev/null 2>&1 ; cd .. ; \
	done

# arXiv submission tarball: flatten common/ into the paper directory,
# include the .bbl, and pack as a gzipped tar.
dist:
	mkdir -p dist

submit-A: A | dist
	@$(MAKE) _submit PAPER=A DIR=$(DIR_A)
submit-B: B | dist
	@$(MAKE) _submit PAPER=B DIR=$(DIR_B)
submit-C: C | dist
	@$(MAKE) _submit PAPER=C DIR=$(DIR_C)
submit-D: D | dist
	@$(MAKE) _submit PAPER=D DIR=$(DIR_D)

_submit:
	@rm -rf dist/paper$(PAPER)
	@mkdir -p dist/paper$(PAPER)
	@cp $(DIR)/main.tex   dist/paper$(PAPER)/
	@cp $(DIR)/main.bbl   dist/paper$(PAPER)/ 2>/dev/null || true
	@cp common/hypertensor.sty dist/paper$(PAPER)/
	@# inline the .sty path in the flattened source
	@sed -i.bak 's|../common/hypertensor|hypertensor|' dist/paper$(PAPER)/main.tex && rm dist/paper$(PAPER)/main.tex.bak
	@cd dist && tar czf paper$(PAPER).tar.gz paper$(PAPER) && echo "  -> dist/paper$(PAPER).tar.gz"
