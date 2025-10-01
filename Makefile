.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

webbrowser.open(sys.argv[1])
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/	
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.DS_Store' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	

clean-test: ## remove test and coverage artifacts	
	rm -fr htmlcov

lint: ## check style with flake8
	flake8 nnetsauce tests

coverage: ## check code coverage quickly with the default Python
	coverage run --source nnetsauce setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate docs		
	pip install autopep8 black pdoc 
	black nnetsauce/* --line-length=80	
	find nnetsauce/ -name "*.py" -exec autopep8 --max-line-length=80 --in-place {} +
	pdoc -t docs nnetsauce/* --output-dir nnetsauce-docs
	find . -name '__pycache__' -exec rm -fr {} +

servedocs: ## compile the docs watching for change	 	
	pip install autopep8 black pdoc 
	black nnetsauce/* --line-length=80	
	find nnetsauce/ -name "*.py" -exec autopep8 --max-line-length=80 --in-place {} +
	pdoc -t docs nnetsauce/* 
	find . -name '__pycache__' -exec rm -fr {} +

release: dist ## package and upload a release
	pip install twine --ignore-installed
	python3 -m twine upload --repository pypi dist/* --verbose

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel	
	ls -l dist

install: clean ## install packages 
	pip install -r requirements.txt 
	black *.py --line-length=80	

run-app: clean ## run the app		
	uv pip install -r requirements.txt	
	#black *.py --line-length=80
	uvicorn main:app --reload

push-stable: ## push stable version
	git add .
	git commit -am "push stable version"
	git push
	git status
