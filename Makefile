.PHONY: setup test run demo template

setup:
	pip install -r requirements.txt

test:
	pytest -v tests/

run:
	python run.py

demo:
	python scripts/run_benchmark.py
	python run.py

template:
	python scripts/make_template.py
