# Status and plan

## Done

- [x] Scaffold, license, requirements.
- [x] `rho_geometry` package: density-matrix construction with validation,
      von Neumann / Renyi entropies, purity, effective rank, PSD square root;
      Uhlmann fidelity, Bures and trace distances, fidelity matrices.
- [x] 44-test suite anchored to analytic values (ln d maximum, purity
      bounds, Bhattacharyya and total-variation classical limits,
      |0>-vs-|+> root fidelity 1/sqrt(2), Bures(orthogonal) = sqrt(2)),
      plus the input-validation contracts of the public API.
- [x] Publication-quality INTRODUCTION and high-school EXTENDED_INTRODUCTION.
- [x] CI on push and pull request, installable package, and a seeded
      script (`analysis/noise_floor.py`) that regenerates the published
      noise-floor table.

## In progress

- [ ] HCP downloader/parcellation (shared with sister repo; Issue #1).
- [ ] State-construction pipeline (Issue #2).

## Planned (Issues #1–#7)

| # | Milestone | Depends on |
|---|---|---|
| 1 | HCP downloader + parcellation (shared) | — |
| 2 | State construction: FC → rho pipeline | 1 |
| 3 | Summary statistics over cohort (entropy, purity, rank) | 2 |
| 4 | H1: task-vs-rest purity/entropy test | 3 |
| 5 | H2: Bures geometry + condition clustering + MDS map | 2 |
| 6 | H3: fidelity fingerprinting day-1 → day-2 | 2 |
| 7 | Null consolidation + figures + write-up | 4–6 |

## Next quarter target

H1 and H3 decided on the full S1200 sample; state-space map published as
interactive figure.
