.PHONY: setup data data-curated data-all explore paper test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	Rscript scripts/setup_r_env.R

data:
	bash scripts/download_hmp.sh
	bash scripts/download_agp.sh

data-curated:
	Rscript scripts/download_curatedmgd.R

data-all: data data-curated

explore:
	jupyter notebook notebooks/01_data_exploration.ipynb

paper:
	cd paper && latexmk -pdf main.tex

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed/* data/results/* figures/*
	cd paper && latexmk -C
