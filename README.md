# density-matrix-brain-geometry

**Treating brain covariance states as formal density matrices — von Neumann
entropy, purity, fidelity and Bures geometry of HCP functional connectivity.**

## Core idea

Any positive-semidefinite matrix `C` with positive trace defines a *formal*
density operator

```
rho = C / tr(C)          (rho >= 0, tr rho = 1)
```

No quantum mechanics in the brain is claimed. The point is *representation*:
normalising a functional-connectivity (FC) matrix this way unlocks a complete,
well-tested vocabulary from quantum information theory — entropy, purity,
fidelity, Bures distance — that turns "how organised is this brain state?"
into geometry with known mathematics and known classical limits.

## Hypotheses

- **H1 (state entropy):** task states have lower von Neumann entropy and
  higher effective-rank concentration than rest (more coordinated = more
  "pure" covariance state).
- **H2 (state distance):** rest and task states are separated in Bures/trace
  distance beyond matched nulls; different tasks cluster by condition.
- **H3 (fingerprint):** a subject's trajectory through state space is
  individually identifiable across scan days (fidelity-matrix matching).

**Falsifier:** if Bures distances between conditions are explained by
trace-norm (total-variance) differences alone, the density-matrix geometry
adds nothing over plain covariance norms and the framework is falsified.

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest tests/ -q        # 17 analytic anchors, all pass
```

```python
import numpy as np
from rho_geometry import (covariance_to_rho, von_neumann_entropy,
                          effective_rank, bures_distance, maximally_mixed)

rng = np.random.default_rng(0)
a = rng.normal(size=(50, 2000))          # 50 channels x 2000 time points
rho = covariance_to_rho(np.corrcoef(a))  # formal density matrix from FC

print(von_neumann_entropy(rho, base=2))  # state "mixedness" in bits
print(effective_rank(rho))               # # of appreciably occupied modes
print(bures_distance(rho, maximally_mixed(50)))
```

## Repository map

| Path | Contents |
|---|---|
| `src/rho_geometry/density.py` | rho construction, von Neumann / Renyi entropy, purity, effective rank, PSD square root |
| `src/rho_geometry/metrics.py` | Uhlmann fidelity, Bures distance, trace distance, fidelity matrices |
| `tests/` | 17 unit tests anchored to analytic values (Bhattacharyya limit, TVD limit, purity bounds, ...) |
| `docs/INTRODUCTION.md` | publication-quality introduction with derivations |
| `docs/EXTENDED_INTRODUCTION.md` | the same ideas for high-school readers, with links |
| `docs/METHODS.md` | HCP data, preprocessing, null models, statistics |
| `docs/STATUS_AND_PLAN.md` | milestone map (mirrors GitHub Issues #1–#7) |
| `docs/CURRENT_RESULTS_AND_DISCUSSION.md` | synthetic validation results, risks, open questions |

## Empirical target

HCP Young Adult S1200: resting-state and 7 task fMRI batteries, Glasser-360
parcellation. See `docs/METHODS.md`.

## License

MIT.
