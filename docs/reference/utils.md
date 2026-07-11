# Utilities Reference

Source: [`practical_classes/utils.py`](https://github.com/das-dias/signal-converters-notebooks/blob/master/practical_classes/utils.py)

Core converter modeling functions shared across all notebooks.

---

## Binary/Decimal Conversion

### `bin2dec(x, width, reverse=False)`

Convert a binary NumPy array (or array of binary arrays) to decimal value(s).

- **x** — Binary array of 0s and 1s, or 2D array of binary words
- **width** — Bit width of each binary word
- **reverse** — If `True`, treat input as LSB-first (default: MSB-first)

### `dec2bin(x, width, reverse=False)`

Convert a decimal value (or array of decimals) to binary NumPy array(s).

- **x** — Integer or array of integers
- **width** — Bit width of the output binary word(s)
- **reverse** — If `True`, output in LSB-first order

---

## DAC Transfer Functions

### `ideal_dac(vref, n_bits)`

Returns a callable `f(Din)` implementing the transfer function of an ideal DAC.

- **vref** — Reference voltage
- **n_bits** — Resolution in bits
- **Returns** — `lambda x: bin2dec(x, n_bits) * vlsb`

---

## ADC Transfer Functions

### `ideal_adc(vref, nbits, roundf)`

Returns a callable `f(Vin)` implementing the transfer function of an ideal ADC.

- **vref** — Reference voltage
- **nbits** — Resolution in bits
- **roundf** — Rounding function (`np.round`, `np.floor`, or `np.ceil`)

### `nonideal_adc(vref, n_bits, ofst=0, gain=1, vnq=0, roundf=np.round)`

Returns `(transfer_function, vtrans)` for a non-ideal ADC with linear errors.

- **ofst** — Offset error voltage
- **gain** — Gain error factor
- **vnq** — Quantization noise voltage level
- **Returns** — Tuple of (callable, transition voltages array)

---

## Binary Arithmetic

### `binadd(a, b)`

Binary word addition. Returns `dec2bin(bin2dec(a) + bin2dec(b), max_width)`.

### `binsub(a, b)`

Binary word subtraction. Returns `dec2bin(bin2dec(a) - bin2dec(b), max_width)`.

---

## Pipeline ADC Support

### `digital_error_correction(douts, scale_factors, reverse=False, bin=True)`

Perform digital error correction for pipeline ADC stages.

- **douts** — List of output code arrays from each pipeline stage
- **scale_factors** — List of scale factors between stages
- **reverse** — Output word bit order
- **bin** — If `True`, return binary; otherwise return decimal
