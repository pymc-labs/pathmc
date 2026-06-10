#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_NAME = pathmc
DOCS_DIR = docs
BUILD_CHECK_DIR = .build-check

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init setup lint check_lint test-fast test docs cleandocs sync-env build check-build help

init: ## Install the package in editable mode without dependencies
	python -m pip install -e . --no-deps

setup: ## Set up the complete development environment
	python -m pip install --no-deps -e .
	python -m pip install -e '.[dev,test,docs,samplers]'
	prek install -f
	@echo "Development environment ready!"

lint: ## Run prek hooks, applying fixes
	prek run --all-files

check_lint: ## Check formatting, linting, and types without making changes
	ruff check .
	ruff format --diff --check .
	mypy --ignore-missing-imports

test-fast: ## Run fast tests, excluding slow MCMC tests
	pytest -x -v -m "not slow"

test: ## Run all tests, including slow integration tests
	pytest -x -v

docs: ## Build the documentation site
	quarto render $(DOCS_DIR)/

cleandocs: ## Clean generated documentation files and Quarto caches
	rm -rf $(DOCS_DIR)/_site $(DOCS_DIR)/_freeze $(DOCS_DIR)/.quarto

sync-env: ## Regenerate environment.yml from pyproject.toml
	prek run pyproject2conda-yaml --all-files

build: ## Build source and wheel distributions
	rm -rf dist build *.egg-info
	python -m build

check-build: build ## Build artifacts and smoke-test sdist and wheel installs
	rm -rf $(BUILD_CHECK_DIR)
	python -m venv $(BUILD_CHECK_DIR)/sdist
	$(BUILD_CHECK_DIR)/sdist/bin/python -m pip install dist/$(PACKAGE_NAME)-*.tar.gz
	$(BUILD_CHECK_DIR)/sdist/bin/python -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__name__)"
	python -m venv $(BUILD_CHECK_DIR)/wheel
	$(BUILD_CHECK_DIR)/wheel/bin/python -m pip install dist/$(PACKAGE_NAME)-*.whl
	$(BUILD_CHECK_DIR)/wheel/bin/python -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__name__)"
	rm -rf $(BUILD_CHECK_DIR)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
