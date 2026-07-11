# Mixed-Signal Converters Course

![Banner](imgs/banner.png)

Educational Jupyter Notebooks for the **Mixed-Signal Converters** course at the Faculty of Sciences and Technology, NOVA University of Lisbon (FCT-NOVA).

This collection covers the systems theory, design, and simulation of **Digital-to-Analog (DAC)** and **Analog-to-Digital Converters (ADC)** using Python.

---

## Practical Classes

| Class | Topic |
|-------|-------|
| [1. Signal Representation](notebooks/practical_class_1.ipynb) | Time/frequency domain, Fourier series, windowing, white noise |
| [2. Ideal ADC/DAC Modeling](notebooks/practical_class_2.ipynb) | Lambda-style transfer functions, binary/decimal conversion |
| [3. Linear & Non-linear Errors](notebooks/practical_class_3.ipynb) | Offset, gain, DNL, INL for DAC and ADC |
| [4. DAC/ADC Error Exercises](notebooks/practical_class_4.ipynb) | SNR computation, non-ideal DAC, spectral analysis |
| [5. Noise Modeling](notebooks/practical_class_5.ipynb) | Jitter, thermal, quantization noise |
| [6. Resistive DAC Modeling](notebooks/practical_class_6.ipynb) | Thermometer, binary weighted, R-2R, Monte Carlo |
| [7. Charge Redistribution DAC](notebooks/practical_class_7.ipynb) | Capacitive DAC, parasitic effects |
| [8. Current Steering DAC](notebooks/practical_class_8.ipynb) | Binary/unitary current steering, statistical analysis |
| [9. Resistive ADC Architectures](notebooks/practical_class_9.ipynb) | Flash, 2-step sub-ranging Flash ADC |
| [10. Capacitive ADC (SAR)](notebooks/practical_class_10.ipynb) | SAR ADC modeling |

## Projects

| Project | Topic |
|---------|-------|
| [Pipeline SAR ADC](notebooks/project1_loopunrolled_pipeline_sar_adc.ipynb) | 2-step loop-unrolled pipeline SAR ADC with digital error correction |

## Getting Started

### Prerequisites

- Python 3.11+ (managed with [uv](https://docs.astral.sh/uv/))
- [just](https://github.com/casey/just) task runner (`brew install just`)

### Installation

```bash
git clone https://github.com/das-dias/signal-converters-notebooks.git
cd signal-converters-notebooks
uv sync
```

### Running Notebooks

```bash
uv run jupyter lab
```

## Key Dependencies

- **NumPy / SciPy** — Signal processing, FFT, window functions
- **Matplotlib + SciencePlots** — Publication-quality plots
- **Seaborn** — Statistical visualizations
