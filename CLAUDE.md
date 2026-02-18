# CLAUDE.md — Project-Wide Rules

## Python Version Constraint: 3.9

**The HPC cluster runs Python 3.9.20. All code must be compatible with Python 3.9.**

### Do NOT use (Python 3.10+ only)
- `X | Y` union type syntax in annotations — use `Union[X, Y]` from `typing` instead
- `X | None` — use `Optional[X]` from `typing` instead

### Preferred approach: drop type annotations
Return type annotations on internal functions are optional and add no runtime value. When in doubt, just omit them rather than adding `Union[...]` boilerplate.

```python
# BAD — crashes on Python 3.9
def setup_augmenter(config: dict) -> Augmenter | None:

# OK — Union from typing
from typing import Optional
def setup_augmenter(config: dict) -> Optional[Augmenter]:

# BEST — just omit it
def setup_augmenter(config: dict):
```

## General Code Rules
- Do not over-engineer. Delete unnecessary abstractions rather than wrapping them.
- Type annotations are optional. Do not add them if they require 3.10+ syntax.
