.PHONY: all

all: paper

midprogress: reports/midprogress/report.pdf

paper: paper/paper.pdf

%.pdf:
	cd $(dir $*) && pdflatex $(notdir $*).tex
	-cd $(dir $*) && bibtex $(notdir $*).aux
	cd $(dir $*) && pdflatex $(notdir $*).tex

clean:
	rm paper/paper.aux paper/paper.bbl paper/paper.blg paper/paper.log \
			paper/paper.pdf reports/midprogress/report.aux \
			reports/midprogress/report.bbl reports/midprogress/report.blg \
			reports/midprogress/report.log reports/midprogress/report.pdf

