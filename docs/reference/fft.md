# FFT Engine Reference

Source: [`practical_classes/fft.py`](https://github.com/das-dias/signal-converters-notebooks/blob/master/practical_classes/fft.py)

Spectral analysis engine for converter output evaluation.

---

## `FFTEngine`

A dataclass that computes the FFT of a signal and extracts spectral performance metrics.

### Constructor

```python
FFTEngine(signal, time=None, unit="V")
```

- **signal** — Input signal array (e.g., ADC/DAC output)
- **time** — Time axis array (optional)
- **unit** — Signal unit label (default: `"V"`)

### Typical Usage

```python
engine = FFTEngine(signal)
engine.fft(n_points=1024, fs=1e6, window="blackmanharris")
engine.compute_harmonics(fin=1e3, fs=1e6, n_harmonics=5)
metrics = engine.spectral_analysis()
```

---

### Core Methods

#### `fft(n_points, fs=None, window="rectangular")`

Compute the power spectrum of the signal.

- **n_points** — Number of FFT points
- **fs** — Sampling frequency
- **window** — Window function: `"rectangular"`, `"hanning"`, `"hamming"`, `"blackmanharris"`, `"blackman"`, `"gaussian"`, `"kaiser"`, `"cosine"`, `"parzen"`

#### `compute_harmonics(fin, fs, n_harmonics=5)`

Identify harmonic bins and power levels from the computed spectrum.

- **fin** — Input signal frequency
- **fs** — Sampling frequency
- **n_harmonics** — Number of harmonics to extract

#### `spectral_analysis()`

Compute all spectral metrics at once. Returns a dictionary with all metrics below.

---

### Metric Getters

Call these after `fft()` and `compute_harmonics()`:

| Method | Returns | Unit |
|--------|---------|------|
| `get_snr()` | Signal-to-Noise Ratio | dB |
| `get_sfdr()` | Spurious-Free Dynamic Range | dB |
| `get_sndr()` | Signal-to-Noise-and-Distortion Ratio | dB |
| `get_thd()` | Total Harmonic Distortion | dB |
| `get_enob()` | Effective Number of Bits | bits |
| `get_h2()` | 2nd Harmonic Distortion | dB |
| `get_h3()` | 3rd Harmonic Distortion | dB |

---

### Supported Windows

| Window | Key |
|--------|-----|
| Rectangular (none) | `"rectangular"` |
| Hanning | `"hanning"` |
| Hamming | `"hamming"` |
| Blackman-Harris | `"blackmanharris"` |
| Blackman | `"blackman"` |
| Gaussian | `"gaussian"` |
| Kaiser | `"kaiser"` |
| Cosine | `"cosine"` |
| Parzen | `"parzen"` |
