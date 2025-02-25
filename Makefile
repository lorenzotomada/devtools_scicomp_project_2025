.PHONY: make
make:
	python -m pip install -r requirements.txt
	python src/pyclassify/utils.py
	
	python -m pip install -e .
.PHONY: clean
clean:
	python -m pip unistall pyclassify -y

