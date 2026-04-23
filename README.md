# gimbal-regression

`gimbal-regression` is a Python package for **Gimbal Regression (GR)** —  
a deterministic local linear regression framework for **stable and reproducible estimation under anisotropic neighborhood geometry**.

The package is designed with a focus on:

- **Deterministic estimation** (no iterative optimization)
- **Numerical stability** under irregular spatial sampling
- **Explicit diagnostics** (conditioning, effective sample size, fallback)
- **Reproducibility** via a single-pass estimator

Unlike conventional local regression methods (e.g., GWR/MGWR), `gimbal-regression` exposes both **estimates and their numerical reliability** as first-class outputs.

The package is distributed on PyPI as `gimbal-regression` and imported in Python as `grpy`.

---

## Installation

### Basic installation

```bash
pip install gimbal-regression
```

### Install from source (development mode)

```bash
git clone https://github.com/yuichiro-otani/gimbal-regression.git
cd gimbal-regression
pip install -e .
```

### Optional Dependencies

Some features require additional packages.
Install as needed:

```bash
# plotting utilities
pip install gimbal-regression[plot]

# benchmarking and comparison methods
pip install gimbal-regression[benchmark]

# development tools
pip install gimbal-regression[dev]

# everything
pip install gimbal-regression[all]
```

## Quick Example

```python
import numpy as np
from grpy import GimbalRegression

rng = np.random.default_rng(42)
n = 100

lat = 35.0 + 0.02 * rng.random(n)
lon = 139.0 + 0.02 * rng.random(n)

x = rng.normal(size=n)
y = 1.0 + 2.0 * x + 0.1 * rng.normal(size=n)

model = GimbalRegression(
    K=20,
    h_m=2000.0,
    gamma=1.0,
)

model.fit(
    y=y,
    x=x,
    lat=lat,
    lon=lon,
)

yhat = model.predict()
diag = model.diagnostics()
summary = model.summary()

print(summary)
```
## Map Visualization

Requires plot extras:
```bash
pip install gimbal-regression[plot]
```
Example:
```python
fig, ax = model.draw_map(
    column="B1",
    title="Local coefficient B1",
    basemap=True,
)
```
## Diagnostics
`grpy` returns diagnostic quantities alongside estimates:
- Condition numbers of local normal matrices
- Effective sample size (ESS)
- Fallback indicators (uniform weighting)

```python
diag = model.diagnostics()
print(diag.head())
```
These diagnostics allow users to directly assess numerical reliability of local estimates, not just predictive accuracy.

## Reproducibility

The estimator is:
- deterministic
- single-pass
- free from stochastic components

This ensures that results are exactly reproducible given identical inputs.

## Project Structure
```
gimbal-regression/
├── src/grpy/
├── tests/
├── examples/
```

- `src/grpy` – core implementation
- `tests/` – unit tests
- `examples/` – usage examples

## Citation

If you use this package, please cite:
```bibtex
@article{Otani2026GR,
  author  = {Otani, Yuichiro},
  title   = {Gimbal Regression: A Geometry-Aware Framework for Stable Local Linear Estimation under Anisotropic Sampling},
  journal = {arXiv preprint arXiv:2603.10382},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.10382}
}
```

## License
MIT License