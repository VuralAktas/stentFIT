# LaTeX Makefile -- auxiliary files go into $(BUILD)

REPORT=Report
BUILD=build

all:    ## Compile the report
	mkdir -p $(BUILD)
	pdflatex -output-directory=$(BUILD) $(REPORT).tex
	-cd $(BUILD) && makeglossaries $(REPORT)
	-cd $(BUILD) && BIBINPUTS=..: bibtex $(REPORT)
	pdflatex -output-directory=$(BUILD) $(REPORT).tex
	pdflatex -output-directory=$(BUILD) $(REPORT).tex
	cp $(BUILD)/$(REPORT).pdf .

clean:  ## Remove all build output
	rm -rf $(BUILD) $(REPORT).pdf
