
<p align="center">
    <img src="docs/imgs/banner.png" alt="msc-nova">
</p>

<h1 align="center">Mixed-Signal Converters Course Notebooks</h1>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Overview

[![made-with-python](https://img.shields.io/badge/Made%20with-Python%203.11-lightgrey)](https://www.python.org/)

This repository contains [Jupyter Notebooks](https://jupyter.org/) for the **Mixed-Signal Converters** course at the Faculty of Sciences and Technology, NOVA University of Lisbon (FCT-NOVA).

The course covers the systems theory, design, and practical aspects of **Nyquist data-rate Digital-to-Analog (DAC)** and **Analog-to-Digital Converters (ADC)**. Content on oversampled converters (sigma-delta modulators, noise shaping) is planned for future additions. The notebooks enable:

- Representing and generating signals in the frequency and time domains
- Modeling random noise (uniform, Gaussian distributions)
- High-level and element-level modeling of DAC and ADC architectures
- Spectral analysis of converter outputs (SNR, SFDR, SNDR, THD, ENOB)

## Directory Organization

```
signal-converters-notebooks/
├── practical_classes/
│   ├── utils.py          # Shared converter modeling utilities
│   ├── fft.py            # FFT spectral analysis engine
│   ├── practical_class_1.ipynb  ... practical_class_10.ipynb
│   ├── project1_loopunrolled_pipeline_sar_adc.ipynb
│   └── project2_c2c_differential_sar_adc.ipynb
├── docs/
│   ├── index.md          # Documentation landing page
│   ├── imgs/             # Circuit diagrams and figures
│   └── reference/        # API reference pages
├── mkdocs.yml            # MkDocs configuration
├── justfile              # Task runner recipes
└── pyproject.toml
```

## Getting Started

### Prerequisites

- **Python 3.11+** managed with [uv](https://docs.astral.sh/uv/)
- **[just](https://github.com/casey/just)** task runner — install via `brew install just`

### Installation

```bash
git clone https://github.com/das-dias/signal-converters-notebooks.git
cd signal-converters-notebooks
uv sync
```

### Running Notebooks

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

## Documentation

The project uses [MkDocs](https://www.mkdocs.org/) with [Material](https://squidfunk.github.io/mkdocs-material/) theme and [mknotebooks](https://github.com/greenape/mknotebooks) to render Jupyter notebooks directly into the documentation site.

```bash
just docs-serve     # Build and serve docs locally at http://localhost:8000
just docs-build     # Build static site into site/
just docs-deploy    # Deploy to GitHub Pages
```

## Quality Checks

```bash
just format         # Ruff-format all notebooks and Python files
just lint           # Ruff lint (use `just lint-fix` for auto-fix)
just pylint         # Pylint notebooks (converts to .py via nbconvert)
just check          # Run all checks together
```

## Dependencies

- [NumPy](https://numpy.org/) — Array processing and numerical computation
- [SciPy](https://scipy.org/) — FFT and signal processing algorithms
- [Matplotlib](https://matplotlib.org/) + [SciencePlots](https://github.com/garrettj403/SciencePlots) — Publication-quality plots
- [Seaborn](https://seaborn.pydata.org/) — Statistical visualizations

## Contributing

Feel free to clone this repository and expand the existing notes. If you discover issues or bugs, please open an [issue](https://github.com/das-dias/signal-converters-notebooks/issues). Changes should be submitted via [pull request](https://github.com/das-dias/signal-converters-notebooks/pulls).

### Main Contributors

- (creator) Diogo André Dias — das.dias@campus.fct.unl.pt

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
