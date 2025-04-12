# Developer notes

## Environment Setup

```bash
uv sync --group dev --extra viz
```

## Dev Checks

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
```
