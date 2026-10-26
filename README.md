# HQ-IST-KdV
Source distribution accompanying *Hybrid Quantum Simulation for Soliton Dynamics of the Korteweg--de Vries System*.
This package implements the positive physical field satisfying

```text
u_t + 6*u*u_x + u_xxx = 0
```

It contains the complete computation, plotting, CSV-export and verification code for the reflectionless single-, double- and triple-soliton benchmarks. Numerical result archives, manuscript files and generated figures are intentionally distributed separately. All required QSP phase resources are included. See [README_CN.md](README_CN.md) for Chinese instructions and [PUBLIC_RELEASE_AUDIT_CN.md](PUBLIC_RELEASE_AUDIT_CN.md) for the file audit and validation record.

## Installation and quick checks

Use Python 3.12 or 3.13. Python 3.12.14 was used for this release audit; the original archive records Python 3.13.5. Run commands from this package directory unless stated otherwise.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\activate.bat` in Windows Command Prompt, then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The source-only test run includes 31 tests and explicitly skips 3 full-data regression tests. For the exact package versions used in the release audit, install `requirements-audit.txt`. `requirements-tested.txt` is the unmodified environment record from the original archive.
