# Contributing to transformers-ocr

Thank you for your interest in contributing! This document covers how to report bugs, suggest features, and submit pull requests.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

---

## Code of Conduct

Be respectful, constructive, and welcoming. Harassment of any kind will not be tolerated.

---

## How to Report a Bug

1. **Search existing issues** before opening a new one.
2. Open a [Bug Report](https://github.com/Lourdhu02/transformers-ocr/issues/new?template=bug_report.md) with:
   - Python and PyTorch version
   - OS and CUDA version (if applicable)
   - Minimal reproduction script
   - Full error traceback

---

## How to Request a Feature

Open a [Feature Request](https://github.com/Lourdhu02/transformers-ocr/issues/new?template=feature_request.md) describing:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

---

## Development Setup

```bash
git clone https://github.com/Lourdhu02/transformers-ocr.git
cd transformers-ocr
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

`requirements-dev.txt` adds `pytest`, `ruff`, and `pre-commit`.

Run pre-commit hooks once:

```bash
pre-commit install
```

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes and add tests where applicable.
3. Run the linter: `ruff check .`
4. Push and open a PR against `main`.
5. Fill in the PR template completely.
6. A maintainer will review within a few days.

PRs should be focused and atomic — one feature or fix per PR.

---

## Style Guide

- **Python**: Follow PEP 8. Line length ≤ 100. Use `ruff` for linting.
- **Docstrings**: Google-style for public functions and classes.
- **Type hints**: Required for all public function signatures.
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, etc.).

---

Thank you for helping make this project better!
