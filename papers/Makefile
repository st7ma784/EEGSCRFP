# Makefile for EEG Causal Hypothesis Paper
# ==========================================
# Usage:
#   make          - Compile paper (full pass)
#   make quick    - Quick compile (no bibliography)
#   make clean    - Remove temporary files
#   make view     - Compile and open PDF
#   make help     - Show this help

.PHONY: help quick main clean view all

# Default target
help:
	@echo "EEG Causal Hypothesis LaTeX Makefile"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  make          - Full compilation with bibliography"
	@echo "  make quick    - Quick compilation (no bibtex)"
	@echo "  make main     - Alias for 'make' (full compilation)"
	@echo "  make clean    - Remove temporary files"
	@echo "  make view     - Compile and open PDF"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make           # Full compile"
	@echo "  make quick     # Quick compile"
	@echo "  make view      # Compile and view PDF"
	@echo "  make clean     # Clean up"

# Full compilation with bibliography
main: clean
	@echo "=== Full LaTeX compilation with bibliography ==="
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "First pass: pdflatex ... done"
	bibtex main.aux > /dev/null 2>&1
	@echo "BibTeX: ... done"
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "Second pass: pdflatex ... done"
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "Third pass: pdflatex ... done"
	@echo ""
	@echo "✓ Compilation complete. Output: main.pdf"

# Alias for main
: main

# Quick compilation without bibliography
quick:
	@echo "=== Quick LaTeX compilation (no bibliography) ==="
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "pdflatex ... done"
	@echo ""
	@echo "✓ Quick compilation complete. Output: main.pdf"
	@echo "Note: References may not be updated. Use 'make main' for full compilation."

# Clean temporary files
clean:
	@echo "=== Cleaning temporary files ==="
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux
	@echo "✓ Temporary files removed"

# Compile and view PDF
view: main
	@echo ""
	@echo "=== Opening PDF viewer ==="
	@if command -v xdg-open > /dev/null; then \
		xdg-open main.pdf & \
	elif command -v open > /dev/null; then \
		open main.pdf & \
	elif command -v start > /dev/null; then \
		start main.pdf & \
	else \
		echo "Could not open PDF. Please open main.pdf manually."; \
	fi

# Watch mode (recompile on any changes)
watch:
	@echo "=== Watching for changes (experimental) ==="
	@echo "Type Ctrl+C to stop"
	@while true; do \
		inotifywait -e modify *.tex sections/*.tex references.bib 2>/dev/null && \
		clear && make quick; \
	done

# Default target is 'main'
.DEFAULT_GOAL := main

# Suppress echoing commands
.SILENT: help quick main view clean watch
