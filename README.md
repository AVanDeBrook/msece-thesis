# Master in Electrical & Computer Engineering Thesis
This repository contains the relevant files for my Master's thesis, including the code, proposal, and thesis documents.

The proposal and thesis are written in latex. Info on building/compiling the LaTeX sources into PDFs can be found below.

Proposal:
```bash
# Build PDF from latex source and clean auxiliary files (build artifacts)
latexmk -gg -bibtex -pdf proposal/MSECE_thesis_proposal.tex && latexmk -c
```
<!--
Thesis:
```bash
# Build PDF from latex source and clean auxiliary files (build artifacts)
latexmk -gg -bibtex -pdf thesis/MSECE_thesis.tex && latexmk -c
```
-->

This is currently a work in progress. I will continue to update this README and repo as I make more progress on my thesis.
