"""Repository path helpers shared by notebooks and tests."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find the nearest parent containing this project's pyproject file."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "README.md").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find the RAG repository root from {current}. "
        "Start inside the repository or pass an explicit path."
    )


def repo_path(*parts: str, must_exist: bool = False) -> Path:
    """Return an absolute repository path and optionally require it to exist."""

    resolved = find_repo_root(Path(__file__)).joinpath(*parts)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Repository path does not exist: {resolved}")
    return resolved
