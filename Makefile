sources = src tests

# SETUP
.PHONY: env
env:
	uv venv

.PHONY: update
update:
	uv pip install -r requirements.txt --upgrade

.PHONY: install
install:
	uv pip install -r requirements.txt

.PHONY: data
data:
	dvc pull

.PHONY: ui
ui:
	fiftyone app launch	

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