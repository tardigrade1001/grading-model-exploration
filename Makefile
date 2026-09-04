# Regenerate every reported number, then verify the write-up against it.
# `make` on its own runs the whole chain in the required order.

PYTHON ?= python

.PHONY: all results figures check test clean

all: check figures

results:
	$(PYTHON) scripts/evaluate.py
	$(PYTHON) scripts/explore.py

# Verifies the markdown against results/, and results/ against a fresh run.
check: results
	$(PYTHON) scripts/check_docs.py --check-results

figures: results
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest

clean:
	rm -rf results figures/results_summary.*
