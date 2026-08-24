#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_NAME = pathmc
BUILD_CHECK_DIR = .build-check
VENV_JUPYTER := $(CURDIR)/.venv/share/jupyter
REFREEZE_QMD_PAGES := $(shell find docs/examples docs/user_guide -name '*.qmd' ! -name '00-welcome.qmd' | sort)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: setup lint check_lint test-fast test jupyter-kernel docs freeze-page refreeze-docs cleandocs build check-build help

setup: ## Set up the complete development environment (uv)
	uv sync --all-extras
	uv run prek install -f
	$(MAKE) jupyter-kernel
	@echo "Development environment ready!"

jupyter-kernel: ## Register the pathmc Jupyter kernel for Quarto execution
	uv run python -m ipykernel install --sys-prefix --name pathmc --display-name pathmc

lint: ## Run prek hooks, applying fixes
	uv run prek run --all-files

check_lint: ## Check formatting, linting, and types without making changes
	uv run ruff check .
	uv run ruff format --diff --check .
	uv run mypy --ignore-missing-imports

TEST_ENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

test-fast: ## Run fast tests, excluding slow MCMC tests
	$(TEST_ENV) uv run pytest -x -v -n auto --dist loadscope -m "not slow"

test: ## Run all tests with coverage, including slow integration tests
	$(TEST_ENV) uv run pytest -x -v -n auto --dist loadscope --cov=pathmc --cov-report=term-missing

docs: jupyter-kernel ## Build the documentation site
	JUPYTER_PATH=$(VENV_JUPYTER) uv run great-docs build

freeze-page: jupyter-kernel ## Re-execute one doc page and refresh its _freeze/ cache (PAGE=docs/examples/.../my_page.qmd)
	@test -n "$(PAGE)" || { echo "Usage: make freeze-page PAGE=docs/examples/01-foundations/my_page.qmd"; exit 1; }
	JUPYTER_PATH=$(VENV_JUPYTER) uv run great-docs freeze "$(PAGE)"
	@echo "Freeze cache refreshed. Commit with: git add _freeze/"

refreeze-docs: jupyter-kernel ## Re-execute all doc notebooks and refresh the committed _freeze/ cache
	JUPYTER_PATH=$(VENV_JUPYTER) uv run great-docs freeze --clean $(REFREEZE_QMD_PAGES)
	JUPYTER_PATH=$(VENV_JUPYTER) uv run great-docs build
	cp -r great-docs/_freeze/index _freeze/
	@echo "Freeze cache refreshed. Commit with: git add _freeze/"

cleandocs: ## Clean the ephemeral great-docs build directory
	rm -rf great-docs

build: ## Build source and wheel distributions
	rm -rf dist build *.egg-info
	uv build

check-build: build ## Build artifacts and smoke-test sdist and wheel installs
	rm -rf $(BUILD_CHECK_DIR)
	uv venv $(BUILD_CHECK_DIR)/sdist
	uv pip install --python $(BUILD_CHECK_DIR)/sdist/bin/python dist/$(PACKAGE_NAME)-*.tar.gz
	$(BUILD_CHECK_DIR)/sdist/bin/python -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__name__)"
	uv venv $(BUILD_CHECK_DIR)/wheel
	uv pip install --python $(BUILD_CHECK_DIR)/wheel/bin/python dist/$(PACKAGE_NAME)-*.whl
	$(BUILD_CHECK_DIR)/wheel/bin/python -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__name__)"
	rm -rf $(BUILD_CHECK_DIR)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
