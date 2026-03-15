.PHONY: setup data data-ibdmdb analyze paper test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	bash scripts/download_agp.sh

data-ibdmdb:
	bash scripts/download_ibdmdb.sh

analyze:
	python scripts/run_agp_bootstrap_v2.py
	python scripts/run_taxa_sensitivity.py
	python scripts/run_ibdmdb_bootstrap.py
	python scripts/run_classification_benchmark.py

explore:
	jupyter notebook notebooks/01_data_exploration.ipynb

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	@echo "PDF ready: paper/main.pdf"

watch:
	@echo "Watching paper/ for changes — press Ctrl-C to stop"
	@while true; do \
	  inotifywait -q -e modify -r paper/ 2>/dev/null && \
	  $(MAKE) paper; \
	done

test:
	python -m pytest tests/ -v

clean:
	rm -rf figures/* results/*.csv
	cd paper && latexmk -C
