# Agent Development Guidelines

This document provides guidelines for AI agents working on this algorithms repository.

## Development Environment

### Nix Shebangs
All Python scripts should use nix shebangs for dependency management:

```python
#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.pyvis -p python313Packages.more-itertools -p python313Packages.click
```

**Benefits:**
- Reproducible builds across different systems
- No need for virtual environments or pip installs
- All dependencies are explicitly declared
- Works seamlessly with the nix package manager

### Required Dependencies
Standard dependencies for most scripts:
- `python313` - Python 3.13 interpreter
- `python313Packages.loguru` - Enhanced logging
- `python313Packages.pytest` - Testing framework
- `python313Packages.click` - CLI framework
- `python313Packages.pyvis` - Network visualization (when needed)
- `python313Packages.more-itertools` - Additional iterator tools

## Code Structure

### Click CLI Framework
All executable scripts should use Click for command-line interfaces:

```python
@click.group()
def cli():
    pass

@cli.command()
@click.option("--param", "-p", default=value, help="Description")
def example(param: type):
    """Command description."""
    # Implementation

if __name__ == "__main__":
    cli()
```

**Benefits:**
- Consistent CLI across all scripts
- Built-in help generation
- Type validation
- Easy testing

### Inline Testing with Pytest
Tests should be defined inline within the same file:

```python
@pytest.mark.parametrize(
    "input, expected",
    [
        (test_case_1, expected_1),
        (test_case_2, expected_2),
    ],
)
def test_function(input, expected):
    result = function(input)
    assert result == expected

@cli.command()
def test():
    pytest.main([__file__])
```

**Benefits:**
- Tests stay close to implementation
- No separate test files to maintain
- Easy to run with `python script.py test`

### Main Functions
Always include a `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    cli()  # or main() function
```

## Code Quality Standards

### Type Annotations
Use abstract types from `collections.abc`:

```python
from collections.abc import Mapping, Sequence, MutableSequence

def function(data: Sequence[float]) -> Mapping[str, int]:
    # Implementation
```

**Prefer immutable types when possible:**
- `Sequence` over `MutableSequence` for read-only data
- `Mapping` over `dict` for read-only mappings
- Only use `MutableSequence` when actually modifying collections

### Code Cleanup
1. **Remove comments** - Code should be self-documenting
2. **Inline once-used variables** - Reduce unnecessary variable assignments
3. **Extract reusable functionality** - Create helper functions for repeated logic
4. **Remove unused imports/variables** - Keep code lean

### Function Signatures
Use modern Python 3.10+ union syntax:

```python
def function(param: str | Path) -> tuple[Sequence[float], Mapping[str, int]]:
    # Implementation
```

## Execution Patterns

### Making Scripts Executable
Before running scripts, make them executable:

```bash
chmod +x path/to/script.py
./path/to/script.py
```

### Running Python Scripts
**CRITICAL:** Always run scripts directly to trigger the nix shebang:

```bash
# Make executable first
chmod +x script.py

# Run directly (this triggers the nix shebang)
./script.py command

# NEVER use python directly - this bypasses nix dependencies
# ❌ python script.py command  # WRONG!
# ❌ python -m module_name     # WRONG!
```

### Testing
Run tests using the built-in test command:

```bash
# Make executable first
chmod +x script.py

# Run tests via the script (triggers nix shebang)
./script.py test

# NEVER use pytest directly - this bypasses nix dependencies
# ❌ pytest script.py  # WRONG!
```

## Visualization

### Pyvis Integration
For graph/network visualizations, use the shared `viz.py` module:

```python
from viz import visualise_graph

visualise_graph(
    graphs=graphs,
    output_filename="output.html",
    graph_titles=titles
)
```

**Features:**
- Multi-graph visualization on same canvas
- Automatic clustering of graph components
- Edge weight labels
- Thematic color schemes
- HTML output with custom styling

### Browser Integration
Include `--open` flags for generated visualizations:

```python
@click.option("--open", is_flag=True, default=False, help="Open in browser")
def example(open: bool):
    # Generate visualization
    if open:
        subprocess.Popen(["open", output])
```

## File Organization

### Directory Structure
- Keep related algorithms in subdirectories
- Use descriptive filenames (e.g., `floyd-warshall.py`, `matrix-method.py`)
- Share common utilities (e.g., `graph.py`, `viz.py`)

### Import Patterns
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from local_module import function
```

## Best Practices

1. **Consistency** - Follow established patterns in the codebase
2. **Simplicity** - Prefer simple, readable solutions
3. **Reusability** - Extract common functionality into shared modules
4. **Testing** - Include comprehensive test cases
5. **Documentation** - Use clear function names and type hints
6. **Performance** - Consider algorithmic complexity and optimization opportunities

## Example Template

```python
#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python313 -p python313Packages.loguru -p python313Packages.pytest -p python313Packages.click
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import click
import pytest

sys.path.append(str(Path(__file__).parent))
from local_module import LocalClass

def algorithm(input_data: Sequence[float]) -> Sequence[float]:
    # Implementation
    return result

@pytest.mark.parametrize("input, expected", [(test_cases)])
def test_algorithm(input, expected):
    assert algorithm(input) == expected

@click.group()
def cli():
    pass

@cli.command()
@click.option("--param", default=5, help="Parameter description")
def example(param: int):
    result = algorithm(range(param))
    click.echo(f"Result: {result}")

@cli.command()
def test():
    pytest.main([__file__])

if __name__ == "__main__":
    cli()
```

**Usage:**
```bash
# Make executable
chmod +x script.py

# Run commands (triggers nix shebang)
./script.py example --param 10
./script.py test
```

This template provides a solid foundation for new algorithms while maintaining consistency with the existing codebase.
