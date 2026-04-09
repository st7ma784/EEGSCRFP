# Research Paper: EEG Reflects Projections of Computational Pathways

This folder contains the academic paper documenting the causal hypothesis test.

## Paper Overview

**Title**: EEG Reflects Projections of Computational Pathways: A Causal Test of Neural Routing via Sparse Attention

**Main Hypothesis**: 
> Internal computational routing structures (reflected in attention patterns) project to observable signals (EEG-like) after dimensionality reduction in a way that preserves information about the routing manipulation.

## Structure

```
papers/
├── main.tex                  # Main paper document
├── references.bib            # Bibliography
├── sections/                 # Individual sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── hypothesis.tex
│   ├── methods.tex
│   ├── methods_attention.tex  # Sparse attention details
│   ├── methods_metrics.tex    # Pathway metrics
│   ├── methods_projection.tex # EEG projection
│   ├── methods_prediction.tex # Predictor architecture
│   ├── experiments.tex        # Experimental design
│   ├── results.tex            # Results and tables
│   ├── discussion.tex         # Discussion and implications
│   ├── limitations.tex        # Limitations and future work
│   ├── conclusion.tex         # Conclusion
│   ├── appendix_metrics.tex   # Detailed metric formulas
│   └── appendix_implementation.tex # Hyperparameters, reproducibility
├── figures/                  # (Empty) Place figures here
└── README.md                 # This file
```

## Compiling the Paper

### Requirements

You need a LaTeX distribution installed:

- **Windows**: MiKTeX or TexLive
- **macOS**: MacTeX
- **Linux**: TexLive

### Compilation Steps

```bash
# Navigate to papers directory
cd papers

# Compile with pdflatex (requires 2-3 runs for references)
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex

# Or use latexmk (automatically handles multiple runs)
latexmk -pdf main.tex

# Or use bibtex for references
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

This produces `main.pdf`.

### Online Compilation

Alternatively, you can compile online without installing LaTeX:

1. Go to [Overleaf](https://www.overleaf.com/)
2. Click "New Project" → "Upload Project"
3. Upload all files from `papers/` folder
4. Click "Recompile"

## Paper Sections

### 1. Introduction & Hypothesis (Sections 1-2)

**Goal**: Motivate the research question and frame the causal hypothesis.

- Why does EEG interpretation matter?
- How do we bridge internal computations and surface recordings?
- What is our specific hypothesis?

**Key Sections**:
- `introduction.tex`: Background and motivation
- `hypothesis.tex`: Formal statement of causal hypothesis

### 2. Methods (Section 3)

**Goal**: Describe the experimental design and all components.

**Key Sections**:
- `methods.tex`: Overview and algorithm
- `methods_attention.tex`: Sparse attention masking (top-k mechanism)
- `methods_metrics.tex`: 6 pathway metrics extracted from attention
- `methods_projection.tex`: Linear transformation to EEG (105 channels)
- `methods_prediction.tex`: MLP architectures and training procedure

### 3. Experiments & Results (Sections 4-5)

**Goal**: Describe the three experiments and present results.

**Experiments**:
1. **Pathway → Sparsity**: Can pathway metrics alone predict sparsity?
2. **EEG → Sparsity**: Can EEG signals predict sparsity?
3. **Transfer**: Do pathway models transfer to EEG predictions?

**Key Sections**:
- `experiments.tex`: Experimental design
- `results.tex`: Results tables and interpretation

### 4. Discussion & Conclusion (Sections 6-7)

**Goal**: Interpret findings and suggest implications.

**Key Topics**:
- Interpretation of pathway-sparsity relationship
- Implications for EEG interpretation
- Comparison to prior work (attention analysis, neural coding)
- Limitations and future directions
- Broader significance

**Key Sections**:
- `discussion.tex`: Detailed discussion
- `limitations.tex`: Limitations and future work
- `conclusion.tex`: Summary and final remarks

### 5. Appendices (A-B)

**Goal**: Provide technical details for reproducibility.

**Key Sections**:
- `appendix_metrics.tex`: Complete metric formulas
- `appendix_implementation.tex`: Hyperparameters, computational cost, reproducibility info

## Key Figures and Tables

### Table 1: Pathway Metrics
Defined in `methods_metrics.tex` and detailed in `appendix_metrics.tex`.

### Table 2: Experiment 1 Results
Pathway metric prediction performance across sparsity levels.

### Table 3: Experiment 2 Results
EEG signal prediction performance.

### Table 4: Experiment 3 Results
Transfer success rate and comparison of pathway vs. EEG predictions.

### Tables 5-6: Baselines and Hyperparameters
Control comparisons and ablations in results section.

## How to Add Figures

1. Save figure files (PNG/PDF) in `figures/` subfolder
2. Add LaTeX code to relevant section:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/your_figure.pdf}
    \caption{Figure caption here.}
    \label{fig:your_label}
\end{figure}
```

3. Reference in text with `\ref{fig:your_label}`

## Citation Format

To cite this paper:

**BibTeX**:
```bibtex
@article{YourName2024,
  title={EEG Reflects Projections of Computational Pathways: 
         A Causal Test of Neural Routing via Sparse Attention},
  author={Your Name and Others},
  journal={Journal Name},
  year={2024}
}
```

**Chicago Style**:
> Your Name, et al. 2024. "EEG Reflects Projections of Computational Pathways: A Causal Test of Neural Routing via Sparse Attention." *Journal Name*.

## Editing the Paper

### Adding New Sections

1. Create new file in `sections/` (e.g., `newSection.tex`)
2. Add to `main.tex`:
   ```latex
   \section{Your Section Title}
   \input{sections/newSection}
   ```
3. Recompile

### Updating Results

Results tables are in `results.tex`. Update numbers directly and recompile.

### Adding References

Add to `references.bib` in BibTeX format, then cite in text with `\cite{key}`.

## Troubleshooting

### "File not found" errors
Make sure you're running `pdflatex` from the papers/ directory:
```bash
cd papers
pdflatex main.tex
```

### References not updating
Run bibtex explicitly:
```bash
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

### PDF not generated
Check for error messages in the log. Common issues:
- Missing `\input{sections/...}` file
- Unmatched `{` or `}`
- References to undefined labels

## Document Statistics

- **Pages**: ~12-15 (depending on figures)
- **Sections**: 8 (including appendices)
- **Equations**: ~15
- **Tables**: 6
- **Figures**: Ready for insertion in `figures/`

## Related Code

This paper documents the implementation in the parent directory:

```
EEGSCRFP/
├── papers/          # ← This directory
├── src/             # Implementation code
├── config/          # Configuration
├── experiments/     # Experiment runners
└── main.py          # Training script
```

To run the experiments that produce the results in this paper:

```bash
cd ..
python main.py --epochs 20 --batch-size 8 --num-prompts 10
```

## Notes for Authors

### Authorship & Acknowledgments

Update the `\author{}` field in `main.tex` with your name(s).

### Funding & Declarations

Consider adding:
- Funding sources
- Conflict of interest statement
- Data/code availability statement

### Future Revisions

Use git to track changes:
```bash
git init
git add papers/
git commit -m "Initial paper draft"
```

## Contact

For questions about the paper or implementation, refer to the main [README.md](../README.md) in the parent directory.

---

**Last Updated**: April 2026  
**Status**: Research Prototype
