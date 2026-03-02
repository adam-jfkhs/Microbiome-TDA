# Future Modules

This directory holds exploratory modules that are **not part of the current paper** and have not been validated against real data.  They are preserved here for potential follow-on work rather than removed entirely.

| Module | Description |
|--------|-------------|
| `src/amr/` | Antimicrobial resistance carrier analysis |
| `src/cirs/` | CIRS biomarker priors and mycobiome decomposition |
| `src/mycobiome/` | Fungal burden and disruption scores |
| `src/neurotransmitter/` | Neurotransmitter pathway potential scoring |

All tests for these modules are in `tests/`.  They run on synthetic data only and are not integrated into the main test suite.

To avoid reviewer confusion, these modules are excluded from the main `src/` tree and from the `pyproject.toml` package discovery scope.  To experiment with them, add `_future/src` to your `PYTHONPATH`.
