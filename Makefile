sources = src tests

# SETUP
.PHONY: env
env:
	# create conda environment from environment.yml
	conda env create -f environment.yml

.PHONY: update
update:
	# update conda environment from environment.yml
	conda env update -f environment.yml

.PHONY: install
install:
	# install package in editable mode so notebooks import src/ package directly
	pip install -e .

.PHONY: data
data:
	# pull data tracked by DVC (requires DVC to be initialised)
	dvc pull


# ANALYSIS AND TESTING
.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --config-file mypy.ini

.PHONY: all
all:
	make format
	make lint
	make mypy

.PHONY: test
test:
	pytest -W ignore::DeprecationWarning  tests