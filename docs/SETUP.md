# Project Setup

This guide establishes one repeatable environment for the learning repository.
The current project requires Python 3.12 or newer; Python 3.12 is the canonical
notebook version while the repository is being standardized.

## 1. Create the environment

From the repository root:

```bash
uv python install 3.12
uv sync --python 3.12
```

Use the environment without manually activating it:

```bash
uv run python --version
```

Expected result: a Python 3.12 interpreter from the project environment.

## 2. Configure credentials

Create a local environment file:

```bash
cp .env.example .env
```

Add only the keys required by the notebook you intend to run. The repository
currently contains lessons that may use OpenAI, Groq, Tavily, or LangSmith.

Callout - Warning:
Never put a real key in a notebook, source file, saved output, or committed
configuration. `.env` is ignored by Git; `.env.example` contains names only.

## 3. Select the notebook kernel

In VS Code, JupyterLab, or another notebook client, choose the interpreter at:

```text
<repository>/.venv/bin/python
```

If a client needs a named Jupyter kernel, register one with:

```bash
uv run python -m ipykernel install --user --name rag-learning-101 --display-name "RAG Learning 101"
```

## 4. Execution rules

- Start notebook sessions from the repository root.
- Run cells top-to-bottom; do not depend on hidden state from an earlier session.
- Use repository-relative paths and the shared path helper once it is added.
- Treat network/provider sections as optional unless the lesson says otherwise.
- Keep saved outputs concise and free of credentials or local absolute paths.

## 5. Validate notebook structure

The validator checks the canonical learning sections, notebook format, saved
errors, suspicious secret prefixes, and machine-specific absolute paths:

```bash
python3 scripts/validate_notebook_structure.py templates/rag-technique-template.ipynb
```

Validate several notebooks at once by passing multiple paths. A non-zero exit
status means at least one notebook failed the checks.

Run canonical offline code cells top-to-bottom:

```bash
python3 scripts/run_notebook_smoke.py 01-rag-foundations/*.ipynb
```

Cells tagged `manual`, `online`, or `paid` are skipped by default. Pass
`--run-all-tags` only when you intentionally want those side effects and have
configured the required services.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Import fails | Wrong interpreter or incomplete sync | Select `.venv/bin/python`, then run `uv sync --python 3.12` |
| API authentication fails | Missing or wrong provider key | Check `.env` and the notebook prerequisites |
| File is not found | Notebook started from another directory | Start from the repository root |
| Results change between runs | Model nondeterminism or unseeded local code | Set supported seeds and compare metrics over multiple cases |
| Notebook is slow on first run | Model/package download | Confirm the download source and allow the first-run setup to finish |

## Setup completion check

- [ ] `uv run python --version` reports Python 3.12.
- [ ] The selected notebook kernel uses the project `.venv`.
- [ ] `.env` exists locally only if credentials are required.
- [ ] The template passes the structure validator.
- [ ] No secret value appears in `git diff`.
