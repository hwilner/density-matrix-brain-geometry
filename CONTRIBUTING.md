# Contributing

1. Pick an open Issue (ordered; check "Depends on").
2. Branch `issue-N-short-name`; commits reference `#N`.
3. New quantities need unit tests anchored to exact analytic values (see
   `tests/test_rho_geometry.py` — classical limits are our favourite
   anchors).
4. Google-style docstrings with the defining formula in each public
   function.
5. `pytest` must be green before PR.
6. Keep the "no quantum-brain claim" phrasing in all user-facing docs.
