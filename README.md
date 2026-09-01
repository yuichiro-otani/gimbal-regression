# gimbal-regression

`gimbal-regression` is a Python implementation of **Gimbal Regression (GR)**, a deterministic local regression framework designed for **inspectable estimation under spatially heterogeneous neighborhood geometry**.

GR constructs local directional weights from two complementary components:

- a **bearing-based reference** derived from neighborhood geometry; and
- a **value-based calibration angle** derived from the local empirical second-moment structure of normalized distance and response.

These quantities modify the local spatial weight metric. They do not represent a stochastic spatial covariance model or a physical geographic axis.

The package is distributed on PyPI as `gimbal-regression` and imported in Python as `grpy`.

---

## Main features

- **Deterministic local estimation**  
  For fixed inputs and parameter settings, GR follows a fixed computational estimator map without iterative parameter optimization.

- **Geometry-aware directional weighting**  
  Local bearing structure is summarized explicitly and used in the directional weight metric.

- **Response-adaptive calibration**  
  The realized directional weights may depend on the observed response through the value-based calibration angle.

- **One-shot ESS safeguard**  
  A deterministic effective-sample-size correction adjusts the local bandwidth when directional weights become concentrated.

- **Uniform fallback**  
  A fallback rule provides a deterministic numerical safeguard when effective support remains insufficient.

- **Inspectable diagnostics**  
  GR exposes quantities describing local orientation, anisotropy, effective sample size, fallback activation, fit quality, and matrix conditioning.

- **Optional benchmarking utilities**  
  The package includes utilities for empirical comparison with several local and spatial prediction methods.

GR is intended for applications in which **explicit local coefficients and target-specific diagnostics** are important. It is not designed as a stochastic spatial-dependence model and does not provide model-based spatial uncertainty quantification.

---

## Installation

### PyPI

```bash
pip install gimbal-regression
```

### From source

```bash
git clone https://github.com/yuichiro-otani/gimbal-regression.git
cd gimbal-regression
pip install -e .
```

### Optional dependencies

Some functionality requires additional packages.

```bash
# plotting utilities
pip install gimbal-regression[plot]

# benchmarking and comparison methods
pip install gimbal-regression[benchmark]

# development tools
pip install gimbal-regression[dev]

# all optional dependencies
pip install gimbal-regression[all]
```

---

## Quick example

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
    n0=15.0,
    min_neff=4.0,
    u_scale=2000.0,
)

model.fit(
    y=y,
    x=x,
    lat=lat,
    lon=lon,
)

print(model.summary())
```

The fitted target-level results are available through:

```python
results = model.results_
print(results.head())
```

---

## Diagnostics

GR returns local diagnostics together with coefficient estimates.

Depending on the requested configuration, these include quantities such as:

- `phi` — bearing-based reference direction
- `r_phi` — bearing-resultant magnitude
- `eta` — geometry-based anisotropy ratio
- `theta_z` — value-based calibration angle
- `neff_raw` — effective sample size before correction
- `neff_post` — effective sample size after the one-shot correction
- `uniform_flag` — uniform-fallback indicator
- `R2` — local coefficient of determination
- `RMSE` — local root mean squared error
- `condM_nor` — condition number of the realized local normal matrix
- `localMoran` — local residual Moran diagnostic, when enabled

For example:

```python
diag = model.diagnostics()
print(diag.head())
```

These quantities are intended to make the **realized local estimator inspectable**, rather than reducing assessment to prediction error alone.

---

## Local coefficients

For a single primary covariate, the local design contains:

1. an intercept;
2. the primary covariate; and
3. a normalized distance-trend term.

Typical coefficient outputs are therefore:

```text
B0    local intercept
B1    local coefficient for the primary covariate
Bz    coefficient for the normalized distance-trend term
```

The distance-trend coefficient is denoted `Bz` in the software output.

---

## Effective-sample-size safeguard

Directional weighting can concentrate local support on a relatively small subset of neighbors.

GR therefore evaluates the effective sample size of the directional weights and applies a one-shot bandwidth correction when the realized ESS falls below the target `n0`.

If effective support remains below `min_neff`, the implementation can activate a uniform-weight fallback.

The relevant diagnostics are exposed explicitly:

```python
results[[
    "neff_raw",
    "neff_post",
    "uniform_flag",
]]
```

`neff_post` is calculated after the one-shot bandwidth correction and before the uniform-fallback decision. It is not the ESS of the final composite regression weights.

---

## Map visualization

Plotting utilities require the optional plotting dependencies:

```bash
pip install gimbal-regression[plot]
```

Example:

```python
fig, ax = model.draw_map(
    column="B1",
    title="Local coefficient B1",
    basemap=False,
)
```

The same interface can be used for diagnostic quantities:

```python
model.draw_map(
    column="eta",
    title="Local anisotropy ratio",
    basemap=False,
)
```

Basemap access is optional. Some external tile providers may require an API key or additional configuration.

---

## Reproducibility

The core GR estimator is deterministic conditional on:

- the input observations;
- their ordering;
- parameter settings;
- package version; and
- the numerical software environment.

The estimator does not use stochastic optimization or random initialization.

For reproducible research, record the installed package version:

```python
from importlib.metadata import version

print(version("gimbal-regression"))
```

For paper replication, it is also useful to record the complete software and hardware environment.

---

## Benchmarking

Optional benchmarking utilities are available through `grpy.benchmark`.

They support empirical comparison of GR with methods including:

- ordinary least squares;
- local ridge regression;
- geographically weighted regression;
- multiscale geographically weighted regression;
- universal kriging; and
- spatial random forest.

Install the required dependencies with:

```bash
pip install gimbal-regression[benchmark]
```

Benchmark results, particularly elapsed runtime, are implementation- and environment-specific and should not be interpreted as hardware-independent algorithmic rankings.

---

## Project structure

```text
gimbal-regression/
├── src/
│   └── grpy/
├── tests/
├── examples/
└── README.md
```

- `src/grpy/` — core estimator, diagnostics, plotting, and benchmarking utilities
- `tests/` — automated tests
- `examples/` — reproducible usage examples and empirical notebooks

---

## Methodological scope

Gimbal Regression is designed as a **deterministic local estimator with explicit computational diagnostics**.

The orientation quantities used by GR should not be interpreted as:

- physical geographic directions inferred from an underlying process;
- principal spatial axes of a stochastic field; or
- parameters of a spatial covariance model.

Instead, they are deterministic summaries used in constructing the realized local directional weights.

Similarly, ESS safeguarding is a numerical support mechanism. A favorable ESS value does not by itself establish full rank or favorable conditioning of the local regression design.

---

## Citation

If you use `gimbal-regression` in research, please cite:

```bibtex
@article{Otani2026GR,
  author  = {Otani, Yuichiro},
  title   = {Gimbal Regression: A Deterministic Estimator Map for Inspectable Local Regression},
  year    = {2026},
  note    = {Manuscript}
}
```

The broader Gimbal Regression manuscript is also available as an arXiv preprint:

```bibtex
@article{Otani2026GimbalRegression,
  author  = {Otani, Yuichiro},
  title   = {Gimbal Regression: Orientation-Adaptive Local Linear Regression under Spatial Heterogeneity},
  journal = {arXiv preprint arXiv:2603.10382},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.10382}
}
```

---

## License

MIT License.