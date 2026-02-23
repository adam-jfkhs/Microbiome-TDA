.PHONY: setup data explore clean paper

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	Rscript scripts/setup_r_env.R

data:
	bash scripts/download_hmp.sh
	bash scripts/download_agp.sh

explore:
	jupyter notebook notebooks/01_data_exploration.ipynb

paper:
	cd paper && latexmk -pdf main.tex

clean:
	rm -rf data/processed/* data/results/* figures/*
	cd paper && latexmk -C
