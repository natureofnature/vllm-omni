# Documentation Build Guide

This directory contains the source files for the vLLM-omni documentation.

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or using requirements file:

```bash
pip install -r requirements/docs.txt
```

### Build and Serve Documentation

From the project root:

```bash
# Serve documentation locally (auto-reload on changes)
mkdocs serve

# Build static site
mkdocs build
```

The documentation will be available at `http://127.0.0.1:8000` when using `mkdocs serve`.

### View Documentation

After building, open `site/index.html` in your browser:

```bash
# On macOS
open site/index.html

# On Linux
xdg-open site/index.html
```

## Auto-generating API Documentation

The documentation automatically extracts docstrings from the code using mkdocstrings. To ensure your code is documented:

1. Add docstrings to all public classes, functions, and methods
2. Use Google or NumPy style docstrings (both are supported)
3. Rebuild the documentation to see changes

Example docstring:

```python
class OmniLLM:
    """Main entry point for vLLM-omni inference.
    
    This class provides a high-level interface for running multi-modal
    inference with non-autoregressive models.
    
    Args:
        model: Model name or path
        stage_configs: Optional stage configurations
        **kwargs: Additional arguments passed to the engine
        
    Example:
        >>> llm = OmniLLM(model="Qwen/Qwen2.5-Omni")
        >>> outputs = llm.generate(prompts="Hello")
    """
```

## Documentation Structure

```
docs/
├── index.md              # Main documentation page
├── getting_started/      # Getting started guides
├── architecture/        # Architecture documentation
├── api/                 # API reference (auto-generated from code)
├── examples/            # Code examples
└── stylesheets/         # Custom CSS
```

## Publishing to Read the Docs

1. Push your changes to GitHub
2. Read the Docs will automatically build the documentation using MkDocs
3. Documentation will be available at: https://vllm-omni.readthedocs.io

## Configuration

The documentation configuration is in `mkdocs.yml` at the project root.

## Tips

- Use `::: module.name` syntax for auto-generated API docs
- Use `--8<-- "path/to/file.py"` for including code snippets
- Use Markdown for all documentation (no need for RST)
- Use Material theme features like admonitions and tabs
