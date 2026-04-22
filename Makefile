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
	python scripts/run_classification_benchmark_v2.py

explore:
	jupyter notebook notebooks/01_data_exploration.ipynb

paper:
	cd paper && latexmk -pdf main.tex

test:
	python -m pytest tests/ -v

clean:
	rm -rf figures/* results/*.csv
	cd paper && latexmk -C
