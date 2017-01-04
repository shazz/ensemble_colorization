.PHONY: all

all: paper

paper: paper/paper.pdf

paper/paper.pdf:
	cd paper && pdflatex paper.tex
	-cd paper && bibtex paper.aux
	cd paper && pdflatex paper.tex

clean:
	rm paper/paper.aux paper/paper.bbl paper/paper.blg paper/paper.log \
			paper/paper.pdf
