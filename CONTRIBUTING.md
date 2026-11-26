# Contributing

Thanks for helping. Ground rules for this repo:

1. **Core algorithms stay hand-written.** numpy is the only runtime dependency;
   no OpenCV, no autograd framework, no QP library. Plotting and serial I/O are
   the exceptions.
2. **Every number in the README comes from a script in this repo.** If you
   change behaviour, re-run `scripts/run_race.py` / `scripts/eval_detector.py`
   and `docs/assets/make_figures.py`, and say on which machine and date.
   Simulated results are marked 🧪; hardware results ⏳ until someone has
   actually flown them.
3. **Tests.** `python -m pytest -q` must pass. Tests that need the SITL library
   skip with a reason when it is not built; CI builds it against skyforge-fc.

## Dev setup

```bash
pip install -e ".[dev,figures]"
python -m pytest -q
python scripts/build_sitl.py --skyforge-fc ../skyforge-fc   # optional: real FC code in the loop
```

Keep `OPENBLAS_NUM_THREADS` small (1-3) on a busy machine; numpy GEMMs collapse
under thread oversubscription.

## Pull requests

Small, focused PRs with a test. Describe how you validated the change. Do not
commit build outputs (`build/`, `*.so`, `*.dll`) or large logs.
