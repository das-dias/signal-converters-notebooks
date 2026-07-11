notebooks := "practical_classes/"

# Format all notebooks and Python files with ruff
format:
    uv run ruff format {{ notebooks }}

# Check formatting without modifying files
format-check:
    uv run ruff format --check {{ notebooks }}

# Lint all notebooks and Python files with ruff
lint:
    uv run ruff check {{ notebooks }}

# Lint and auto-fix what ruff can
lint-fix:
    uv run ruff check --fix {{ notebooks }}

# Pylint all notebooks (converts to .py via nbconvert, then lints)
pylint:
    #!/usr/bin/env bash
    set -euo pipefail
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT
    for nb in {{ notebooks }}*.ipynb; do
        name=$(basename "$nb" .ipynb)
        uv run jupyter nbconvert --to script --output-dir "$tmpdir" "$nb" 2>/dev/null || continue
        echo "==> pylint: $nb"
        uv run pylint --disable=C,R,W0621,W0612,E0401,E1101,W0104,W0611,E0611,E0602 "$tmpdir/${name}.py" || true
    done

# Pylint Python utility modules directly
pylint-utils:
    uv run pylint {{ notebooks }}utils.py {{ notebooks }}fft.py

# Run all checks: format check, lint, and pylint
check: format-check lint pylint

# --- Documentation ---

# Prepare docs: copy notebooks and support files into docs/notebooks/
docs-prepare:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p docs/notebooks
    for nb in {{ notebooks }}*.ipynb; do
        # skip empty/invalid notebooks
        [ -s "$nb" ] || continue
        python -c "import json, sys; json.load(open(sys.argv[1]))" "$nb" 2>/dev/null || continue
        cp "$nb" docs/notebooks/
    done
    cp {{ notebooks }}utils.py {{ notebooks }}fft.py docs/notebooks/
    # copy images so notebook-relative paths resolve
    cp -r docs/imgs docs/notebooks/imgs 2>/dev/null || true

# Build documentation site
docs-build: docs-prepare
    uv run mkdocs build

# Serve documentation locally
docs-serve: docs-prepare
    uv run mkdocs serve

# Deploy documentation to GitHub Pages
docs-deploy: docs-prepare
    uv run mkdocs gh-deploy --force

# Clean generated docs artifacts
docs-clean:
    rm -rf docs/notebooks site
