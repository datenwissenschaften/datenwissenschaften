# AGENTS.md

# Purpose

This project follows a strict engineering philosophy centered around:

- KISS (Keep It Simple, Stupid)
- DRY (Don't Repeat Yourself)
- YAGNI (You Aren't Gonna Need It)
- SOLID where it improves clarity
- Explicit over implicit
- Composition over inheritance
- Extreme Programming (XP)
- Readability over cleverness
- Prefer mature external libraries over custom implementations

---

# Major Versions

Major versions are clean breaks.

- Remove superseded code instead of deprecating it.
- Never add compatibility aliases, legacy import paths, migration layers, or fallback behavior.
- Never read old configuration shapes, persisted schemas, or checkpoint formats.
- Reject unsupported input explicitly and fail fast.
- Update every internal caller to the current API in the same change.

---

# Extreme Programming (XP)

Apply the principles of Extreme Programming throughout the project.

## Simplicity

Implement the simplest solution that correctly solves the current problem.

Do not design for hypothetical future requirements.

Never add documentation or comments. The code has to be self-explanatory.

Remove legacy documentation and comments.

---

## Small Iterations

Implement functionality in small, incremental changes.

Every change should leave the codebase in a working state.

---

## Continuous Refactoring

Improve the design whenever code is touched.

Remove duplication.

Simplify abstractions.

Improve naming.

Reduce complexity.

---

## Collective Code Ownership

Write code that any developer can understand and modify.

Avoid personal coding styles.

Prefer consistency over preference.

---

## Coding Standards

The entire codebase should follow one consistent style.

All code must pass:

- `black --check`
- `ruff check`
- `ruff format --check`

---

## Continuous Integration

Every commit should leave the project buildable.

Broken code must never be committed.

---

## Feedback

Prefer fast feedback.

Fail fast.

Raise exceptions instead of silently ignoring problems.

---

## Sustainable Pace

Optimize for maintainability rather than short-term speed.

---

# External Libraries First

Before writing custom code, always check whether a mature external library already solves the problem.

Preferred libraries include:

| Purpose | Library |
|----------|----------|
| configuration | python-box |
| validation | pydantic |
| CLI | typer |
| HTTP | httpx |
| retries | tenacity |
| logging | loguru |
| serialization | msgspec |
| YAML | pyyaml |
| caching | cachetools |
| async files | aiofiles |
| console output | rich |

Do not reinvent existing solutions.

---

# Code Organization

## Small Classes

- Maximum **100 lines** per class.
- One responsibility per class.
- Split large classes into focused components and files.
- Place exactly one class in each Python file.
- Move each extracted class into a feature-specific submodule.
- Do not re-export moved classes from their previous modules.

---

## Small Functions

- One responsibility.
- Prefer early returns.
- Avoid deep nesting.
- Keep functions concise.

---

## Feature-Based Structure

Organize by feature instead of technical type.

- Keep only `__init__.py` in the `datenwissenschaften` package root.
- Place every implementation file in a feature-specific subpackage.
- Name packages and modules after their domain responsibility.
- Colocate framework integrations with the domain they serve. For example, metadata callbacks belong in `metadata`, not `callbacks`.
- Never create generic technical packages such as `callbacks`, `core`, `helpers`, `utils`, `misc`, or `common`.

Good

```text
training/
    agent.py
    reward.py
    state.py
```

Avoid

```text
helpers/
utils/
misc/
common/
```

---

## No Logic in `__init__.py`

Never place executable code inside `__init__.py`.

Do not import modules solely for side effects.

Keep `__init__.py` empty and never use it to re-export names.

---

# Function Design

## No Default Parameters

Never use default parameter values.

Bad

```python
def load(path=None):
```

Good

```python
def load(path):
```

Every caller must explicitly provide every argument.

---

## No Hidden Defaults

Never silently substitute missing values.

Avoid

```python
config.get("host")
config.get("host", None)

getattr(obj, "value")
getattr(obj, "value", None)
```

Prefer explicit checks.

```python
if "host" in config:
    host = config["host"]

if hasattr(obj, "value"):
    value = obj.value
```

Missing values should normally raise an exception.

---

# Python Style

## Type Hints

Every function, method and public attribute should be typed.

---

## Dataclasses

Prefer

```python
@dataclass(slots=True)
```

for immutable or data-only objects.

---

## Path Handling

Always use `pathlib`.

Never use `os.path`.

---

## Imports

- No wildcard imports.
- Group imports logically.
- Avoid local imports unless solving circular dependencies.

---

## Naming

Prefer descriptive names.

Good

```python
learning_rate
```

instead of

```python
lr
```

unless universally accepted.

---

## Logging

Always use `loguru`.

Never use `print()`.

---

## Configuration

Load configuration once.

Prefer YAML.

Use `python-box` for configuration access.

Example

```python
config.database.host
```

instead of

```python
config["database"]["host"]
```

---

## Error Handling

Fail fast.

Catch only expected exceptions.

Never write

```python
except Exception:
    pass
```

---

## Constants

Avoid magic numbers and repeated literals.

Extract named constants.

---

## Comments

Code explains *how*.

Comments explain *why*.

Delete commented-out code.

---

# Design Principles

- KISS
- DRY
- YAGNI
- SOLID where appropriate
- Composition over inheritance
- Explicit dependencies
- Immutable data where practical
- Pure functions whenever possible

---

# Code Quality

## Keep It Concise

Write the fewest lines necessary while remaining readable and maintainable.

Avoid unnecessary variables, wrappers and abstractions.

---

## Readability First

Readable code is always preferred over clever code.

Optimize for the next developer.

---

## Refactor Frequently

Whenever code is modified:

- simplify it
- reduce duplication
- improve naming
- reduce complexity
- leave it cleaner than before

---

# Formatting and Linting

Every change must pass:

```bash
black --check src
ruff check src
ruff format --check src
```

Black and Ruff must agree on formatting. Follow their recommendations unless there is a compelling architectural
reason not to.

---

# Final Checklist

Before considering work complete, verify:

- Simpler solution not available
- No duplicated logic
- No unnecessary abstractions
- No default parameters
- No hidden defaults
- No logic in `__init__.py`
- Classes under 100 lines
- Functions are small and focused
- Uses mature external libraries where appropriate
- Fully typed
- Uses `pathlib`
- Uses `python-box`
- Uses `loguru`
- Passes `black --check`
- Passes `ruff check`
- Passes `ruff format --check`
- Leaves the codebase cleaner than it was found
