# Development Guide

This guide provides information for developers working on vLLM-omni.

## Setting Up Development Environment

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

## Running Tests

Run all tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=vllm_omni --cov-report=html
```

## Building Documentation

Build documentation locally:

```bash
mkdocs serve
```

Build static site:

```bash
mkdocs build
```

## Code Style

We use the following tools for code style:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all style checks:

```bash
black vllm_omni tests
isort vllm_omni tests
flake8 vllm_omni tests
mypy vllm_omni
```

## Project Structure

```
vllm_omni/
├── core/           # Core scheduling and caching
├── engine/         # Engine components
├── entrypoints/    # User-facing APIs
├── inputs/         # Input processing
├── model_executor/ # Model execution
├── outputs.py      # Output handling
└── worker/         # Worker processes
```

## Writing Documentation

- Use Markdown for documentation
- Include docstrings for all public APIs
- Follow the existing documentation style
- Update API documentation when adding new features

