# Master in Electrical & Computer Engineering Thesis
This repository contains the relevant files for my Master's thesis, including the code, proposal, and thesis documents.

File structure info:
* `data_utils` - Folder for relevant data processing scripts (written in languages other than Python)
* `proposal` - Root folder for the thesis proposal LaTeX sources, figures, and any other graphics
* `source` - Root folder for all data processing, analysis, model training, testing, analysis, and testing noteboooks
* `thesis` - Root folder for thesis LaTeX, BibTeX, and figure sources

The proposal and thesis are written in latex. Info on building/compiling the LaTeX sources into PDFs can be found below.

# Building the LaTeX sources
Proposal:
```bash
cd proposal
# Build PDF from latex source and clean auxiliary files (build artifacts)
latexmk -gg -bibtex -pdf MSECE_thesis_proposal.tex && latexmk -c
```

Thesis:
```bash
cd thesis
# Build PDF from latex source and clean auxiliary files (build artifacts)
latexmk -gg -bibtex -pdf MSECE_thesis.tex && latexmk -c
```

# Setting up the Python/Conda environment to run the Python sources
It is highly recommended to use Anaconda/Miniconda3 to manage the Python environment and packages.

Create the Conda environment with the following:
```bash
# create python 3.9 environment
conda create -n thesis python=3.9
# activate environment
conda activate thesis
```

Install PyTorch using conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then install the required packages, using `pip`, from the `requirements.txt` file in the root of the repository:
```bash
pip install -r requirements.txt
```

This is currently a work in progress. I will continue to update this README and repo as I make more progress on my thesis.
