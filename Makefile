.DEFAULT_GOAL := help # default behaviour

help:
	@echo "To perform tests and format the code properly, activate the desired environment and type 'make all'"

tests:
	@echo "Running tests with pytest..."
	python -m pip install .
	python -m pytest -v

docs:
	@echo "Building documentation..."
	cd docs
	make html
	cd ..

format:
	@echo "Formatting code with black..."
	python -m black .

all: tests docs format
	@echo "Tests completed, documentation generated, and code formatted."