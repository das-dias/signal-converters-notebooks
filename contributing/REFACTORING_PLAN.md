# Refactoring Plan: All Notebooks and Utility Files

## Context

All 12 notebooks in this mixed-signal converters course share systemic issues: raw HTML instead of Markdown, hundreds of unnecessary code comments, critical explanatory comments trapped in code cells, typos, deprecated APIs, unused imports, and several high-severity bugs. The utility files (`utils.py`, `fft.py`) have dead code, wrong docstrings, and naming issues. This refactoring adopts a scientific, self-documenting style across the entire repository — concise markdown, proper LaTeX, clean code.

**Scope:** `utils.py`, `fft.py`, all 10 practical class notebooks, `project1_loopunrolled_pipeline_sar_adc.ipynb`. `project2_c2c_differential_sar_adc.ipynb` is an empty (0-byte) file — skip it.

---

## Phase 1: Utility File Cleanup — `utils.py`

**File:** `practical_classes/utils.py`

All changes are backward-compatible. No public function signatures change except renaming `bin` → `as_binary` in `digital_error_correction` (all call sites are commented out — verified by grep).

| Change | Lines | Detail |
|--------|-------|--------|
| Remove `import pdb` | 3 | Unused; all `pdb.set_trace()` calls are in comments |
| Remove commented-out assert | 19 | Dead code |
| Fix lambda shadowing in `bin2dec` | 20, 24 | `lambda x,b:` → `lambda acc, bit:` to avoid shadowing outer `x` |
| Remove redundant `np.array()` wrapping | 24 | `np.array(np.array([...]))` → `np.array([...])` |
| Remove `# define the function to convert...` comment | 5 | Obvious |
| Fix `bin2dec` docstring | 7-14 | Replace nonexistent `b` param with actual params `x`, `width`, `reverse` |
| Fix `dec2bin` docstring | 27-34 | Same fix |
| Fix `ideal_adc` docstring | 57-64 | Change `n_bits` → `nbits` to match actual parameter name |
| Fill `binsub`/`binadd` docstring placeholders | 113-115, 125-127 | Replace `_type_`/`_description_` with actual param descriptions |
| Rename `bin` → `as_binary` in `digital_error_correction` | 138, 185 | Avoids shadowing Python built-in |
| Remove commented-out dead code block | 154-167 | ~14 lines of abandoned logic |
| Remove commented-out `pdb.set_trace()` calls | 97, 171, 175, 178 | Leftover debugging |

---

## Phase 2: Utility File Cleanup — `fft.py`

**File:** `practical_classes/fft.py`

| Change | Lines | Detail |
|--------|-------|--------|
| Remove unused imports: `pdb`, `os`, `traceback` | 3, 5, 6 | None are referenced |
| Remove `@dataclass` decorator | 33 | No-op — class has manual `__init__` |
| Rename `np` → `n` in `windows` dict lambdas | 22-30 | `lambda x, np:` shadows the numpy alias; use `lambda x, n:` |
| Rename `bin` → `freq_bin` in `compute_harmonics` loop | 165-168, 171 | Shadows Python built-in |
| Fix `fft` docstring: replace `_summary_`, remove nonexistent `signal` param | 77-80 | Placeholder + ghost param |
| Fix `get_h3` docstring | 318 | Says "second harmonic" — should say "third harmonic" |
| Remove stale `span` param from `get_h2`/`get_h3` docstrings | 304-307, 319-320 | Neither method takes `span` |
| Fix duplicated `Args:` in `get_enob` | 335 | Two `Args:` headers |
| Fix copy-paste assert messages | 231-232, 282-283, 309-311, 322-324, 341-342 | All say "SFDR" — fix to match actual metric |
| Remove unused `unit` param from `__init__` | 41 | Never stored or used; no notebook passes it |
| Delete empty `SpecificationTester` class | 394-396 | Placeholder with `pass` |

---

## Phase 3: Notebook Refactoring — Common Patterns

These patterns apply to **every notebook** and should be applied systematically:

### 3a. HTML → Markdown conversion (all notebooks)

Every notebook uses raw HTML (`<h1 align="center">`, `<p align="justify">`, `<ul>/<li>/<il>`) for headings, paragraphs, and lists. Replace all with standard Markdown:
- `<h1 align="center">Title</h1>` → `# Title`
- `<h2>Section</h2>` → `## Section`
- `<p align="justify">Text</p>` → plain paragraph text
- `<ul><li>...</li></ul>` → Markdown bullet lists
- Fix `<il>` (wrong tag) → proper Markdown lists (affects notebooks 9, project1)

### 3b. Remove unnecessary/obvious code comments (all notebooks)

Estimated ~400+ total across all notebooks. Categories to remove:
- **"Define" comments**: `# define the signals`, `# define the DAC`, `# define the time base`
- **"Plot" comments**: `# plot the results`, `# visualize the signals`, `# observe the spectrum`
- **"Compute" comments**: `# compute the quantization error`, `# compute the output codes`
- **Duplicate header comments**: `# spectral analysis using fft`, `# frequency domain analysis`

### 3c. Keep unit annotations

Inline `# [V]`, `# [V^2]`, `# [dB]`, `# [Hz]`, `# [F]`, `# [C]` are compact and useful — keep them.

### 3d. Pull critical comments to markdown cells

Insert a new markdown cell with properly formatted content before code cells that contain explanatory comments about theory, design rationale, or pedagogical context. Delete the comment from the code cell.

### 3e. Remove dead code across all notebooks

- Commented-out `pdb.set_trace()` calls
- Commented-out assertions
- Commented-out `print` statements
- Commented-out `sns.distplot()` calls
- Commented-out `#use_line_collection=True`

### 3f. Clean up unused imports (all notebooks)

Many notebooks import 5 functions from `utils.py` but use only 1-2. Remove unused imports. Also remove unused `scipy`, `scipy.signal`, `uniform`, `partial` imports.

### 3g. Consolidate mid-notebook imports

Move any mid-notebook imports to the first code cell (affects notebooks 1, 4, 6, project1).

### 3h. Fix Portuguese placeholder text

Notebooks 9 and 10 contain "INSERIR IMAGEM..." TODO placeholders in Portuguese. Replace with English placeholders or descriptive markdown (e.g., `> **TODO:** Insert circuit diagram of the flash ADC architecture.`).

### 3i. Fix typos (all notebooks)

Comprehensive list across all notebooks — fix during the per-notebook pass. Major recurring ones: "votlage"→"voltage", "represenation"→"representation", "supperposed"→"superposed", "wheight"→"weight", "limitted"→"limited", "memmory"→"memory", "covnersion"→"conversion", "becasue"→"because".

---

## Phase 4: Per-Notebook Refactoring Details

### Notebook 1 — Signal Representation

**File:** `practical_classes/practical_class_1.ipynb`

**Delete:** Cells 23-25 (nbconvert, %mv, empty cell)

**Bug fixes:**
| Cell | Bug | Fix |
|------|-----|-----|
| 6 | `phase2_rad = np.deg2rad(phase1)` uses wrong variable | Change to `np.deg2rad(phase2)` |
| 6 | `x2 = ... + phase1_rad` uses wrong phase | Change to `phase2_rad` |
| 5, 7 | `ax.legend()` called with no labels | Add `label=r"$x_1(t)$"` etc. |
| 21 | `sns.distplot()` deprecated | Replace with `sns.histplot(..., kde=True, stat="density")` |

**Equation fixes (Cell 11 — differential signaling):**
- Fix power-series nonlinearity model to standard textbook form: single-tone input through a weakly nonlinear system, producing harmonics at multiples of the fundamental
- Show differential output cancels even-order terms
- Replace `*` with `\cdot`, fix `180 º` → `180^\circ`
- Define or remove the unexplained DC term

**Equation fixes (Cell 19 — noise):**
- Replace `\hspace{10pt} (1)` numbering with `\tag{1}`

**Critical comments → markdown cells (6 new markdown cells):**
- Simulation time base rationale + MATLAB↔NumPy equivalence
- Harmonics definition (why `x2`/`x3` are redefined)
- FFT normalization (currently repeated 3× in code comments — explain once)
- Uniform noise model for quantization
- Gaussian noise motivation (CLT)
- Signal + noise superposition purpose

**Self-documenting improvements:**
- Use `f2 = 2 * f1` instead of `f2 = 2e3` to make harmonic relationship explicit
- Remove all trailing bare-expression lines

---

### Notebook 2 — Ideal DAC/ADC

**File:** `practical_classes/practical_class_2.ipynb`

- **8 cells total** (3 md, 5 code) — smallest notebook
- Fix typo "covnersion"
- ~10 unnecessary comments to remove
- 1 critical comment to pull to markdown: DAC step function behavior
- **Replace local function copies with imports:** This notebook duplicates `bin2dec`, `dec2bin`, `ideal_dac`, `ideal_adc` locally. Delete the local definitions and add `from utils import bin2dec, dec2bin, ideal_dac, ideal_adc` to the imports cell. This avoids divergence between the notebook and utils.py.

---

### Notebook 3 — Linear and Non-Linear Errors (DNL/INL)

**File:** `practical_classes/practical_class_3.ipynb`

- 22 cells (8 md, 14 code)
- Fix typos: "voltlallges", "exagerated", "procees"
- ~45 unnecessary comments
- ~8 critical comments → markdown (especially multi-line ADC gain error explanations in cell 13)
- **HIGH BUG:** Cell 15 uses DAC variable `vout_off_eos` in the ADC section — conceptually wrong; should use ADC outputs

---

### Notebook 4 — Exercises on DAC/ADC and Error Modeling

**File:** `practical_classes/practical_class_4.ipynb`

- 22 cells (5 md, 17 code)
- Mix of HTML and Markdown (only ~20% HTML)
- ~28 unnecessary comments
- Fix typos: "aleady", "characterisitic", "covnerges"
- **HIGH BUG:** Cell 19 — `ffactor` undefined variable → `NameError` crash. Should be `f_scale`
- **HIGH BUG:** Cell 18 — `normal(0, vn**2, Np)` passes variance where std dev is expected → incorrect noise model
- **HIGH BUG:** Cell 12 — offset formula uses `+` instead of `-` (accidentally correct only because `vout_ideal[0]==0`)
- Move `%matplotlib inline` from cell 6 to cell 1

---

### Notebook 5 — Noise Modeling

**File:** `practical_classes/practical_class_5.ipynb`

- 31 cells (11 md, 20 code)
- Heavy HTML (~82% of markdown cells)
- Fix typos: "aproximated", "Innevitable", "juustify", "reuslting", "independant", "soruces", "votlage"
- **HIGH BUG:** Cell 7 — jitter noise double-applies std dev: `jitter_stdev * normal(0, jitter_stdev, Np)` gives effective std dev = `jitter_stdev²`. Should be just `normal(0, jitter_stdev, Np)`
- **CONTENT BUG:** Cell 3 says "Analog-to-Analog converter" — should be "Analog-to-Digital"
- ~40 unnecessary comments, ~5 critical comments → markdown

---

### Notebook 6 — Resistive DAC Element-Level Modeling

**File:** `practical_classes/practical_class_6.ipynb`

- 37 cells (10 md, 27 code)
- ~30 unnecessary comments, ~5 critical comments → markdown
- Fix typos: "wheight" (×6 occurrences), "stelling", "respose", "agains", "trasnfer", "vlasb", "innevitably", "subsequnetly", "iohms", "wrost"
- Remove `import pdb` from cell 29
- Remove unused imports: `ideal_dac`, `ideal_adc`, `nonideal_adc`, `bin2dec` (only `dec2bin` used)
- Remove unused `scipy`, `scipy.signal`, `uniform`
- **BUG:** Cell 10 — `ax.set_xticklabels(tD/1e-9)` passes a 1M-element array as tick labels
- **Comment/value mismatch:** Cell 20 says `# 15%` but value is `0.05` (5%)
- Cell 17 has a multi-line triple-quoted string acting as a comment — convert to markdown

---

### Notebook 7 — Charge Redistribution DAC

**File:** `practical_classes/practical_class_7.ipynb`

- 18 cells (3 md, 15 code) — relatively small
- Fix typos: "jusitfy", "limitted"
- ~15 unnecessary comments, ~3 critical comments → markdown
- Remove unused imports: `ideal_dac`, `ideal_adc`, `nonideal_adc`, `bin2dec` (only `dec2bin` used)
- Remove unused `scipy`, `scipy.signal`, `uniform`

---

### Notebook 8 — Current Steering DAC Statistical Analysis

**File:** `practical_classes/practical_class_8.ipynb`

- 27 cells (4 md, 23 code)
- Fix typos: "nefast" (Portuguese/English confusion), "signle", "retreive"
- ~25 unnecessary comments, ~3 critical comments → markdown (Monte Carlo section headers)
- Remove unused imports: `ideal_dac`, `ideal_adc`, `nonideal_adc` (only `dec2bin`, `bin2dec` used)

---

### Notebook 9 — Resistive ADC Architectures (Flash, 2-Step Flash)

**File:** `practical_classes/practical_class_9.ipynb`

- 37 cells (8 md, 29 code)
- Heavy HTML with `<il>` typo instead of `<li>` (2 cells)
- **Portuguese placeholders:** 2 cells with "inserir imagem..." TODO text
- Fix typos: "votlage", "enphasis", "ofset", "becasue", "recursivelly", "constannt", "covnersion", "limitted", "inicialize", "threshhold"
- ~50 unnecessary comments, ~10 critical comments → markdown
- **BUG:** `dac_linearity` variable name in an ADC context (cell 13) — rename to `adc_linearity`
- ~38 lines of commented-out code, especially cell 33 (~25 lines)
- Remove unused imports: all 5 from utils are unused; `scipy`, `scipy.signal`, `partial`, `uniform` unused
- **LaTeX issues (Cell 1):** Missing braces in subscript; `W.L` → `W \cdot L`; missing exponent on squared term

---

### Notebook 10 — Capacitive ADC (SAR)

**File:** `practical_classes/practical_class_10.ipynb`

- 41 cells (6 md, 35 code)
- **Portuguese placeholders:** 9 instances of "INSERIR IMAGEM..." across cells 9, 17, 26
- Fix typos: "supperposed", "mponte", "aditional" (×4), "DIFEERENCIAL", "MONOTONICO"
- ~80+ unnecessary comments — heaviest of all notebooks
- ~15 critical comments → markdown (especially SAR algorithm explanations in cells 11, 20, 29)
- **Code duplication:** Cells 11, 20, 29, 38 are near-identical SAR simulation loops (~80-100 lines each) — keep separate for pedagogical step-by-step readability, but add a concluding markdown section explaining how these loops could be consolidated into a shared function
- **BUG:** Cell 38 has `VERBOSE = True` in a 1000-iteration Monte Carlo loop — will flood output
- Empty markdown sections (cells 17, 35 have `<p align="justify">` with no content)
- Remove unused `dec2bin` import (only `bin2dec` used)

---

### Project 1 — Loop-Unrolled Pipeline SAR ADC

**File:** `practical_classes/project1_loopunrolled_pipeline_sar_adc.ipynb`

- 51 cells (12 md, 39 code) — largest notebook
- Heavy HTML with `<il>` typo instead of `<li>` (3 cells)
- Fix typos: "assynchronous", "encharged", "splitted", "aditional", "settlinf", "virst", "mponte", "transistion", "writting"
- ~40-50 unnecessary comments
- ~8-12 critical comments → markdown (algorithm phase headers, multi-line NOTEs)
- **BUG:** Cell 34 — wrong INL plotted for 2nd sub-ADC: plots first sub-ADC data with "2nd sub ADC" title
- **BUG:** Cell 33 — label has stray leading `f` character
- Remove `import pdb` (cell 4, unused)
- Move `from fft import FFTEngine` from cell 48 to cell 4
- Remove unused imports: `ideal_dac`, `ideal_adc`, `nonideal_adc`, `dec2bin`, `digital_error_correction` (only `bin2dec` used)

---

## Phase 5: Verification

Execute after each notebook commit to catch regressions early:

1. **`just check`** — ruff format-check, lint, pylint
2. **Kernel restart + run-all** on every modified notebook — verify:
   - All plots render correctly
   - No deprecation warnings
   - No `NameError` or `UserWarning`
3. **`just docs-build`** — verify MkDocs renders all notebooks correctly

---

## Commit Strategy

Execute in order, one commit per unit of work:

1. `fix(utils): clean docstrings, remove dead code, fix variable shadowing`
2. `fix(fft): fix docstrings, remove dead imports, fix copy-paste errors`
3. `refactor(class-1): rewrite markdown, fix equations and bugs, adopt self-documenting code`
4. `refactor(class-2): convert HTML to markdown, replace local function copies with imports`
5. `refactor(class-3): convert HTML, fix ADC error computation bug, clean comments`
6. `refactor(class-4): fix undefined ffactor, noise model bugs, convert HTML, clean comments`
7. `refactor(class-5): fix jitter noise bug, convert HTML, fix typos, clean comments`
8. `refactor(class-6): convert HTML, fix tick labels bug, remove pdb, clean comments`
9. `refactor(class-7): convert HTML, clean comments, remove unused imports`
10. `refactor(class-8): convert HTML, clean comments, remove unused imports`
11. `refactor(class-9): fix HTML il tags, LaTeX, Portuguese placeholders, clean comments`
12. `refactor(class-10): fix Portuguese placeholders, add SAR consolidation guide, clean comments`
13. `refactor(project-1): fix INL plot bug, label typo, convert HTML, clean comments`

---

## Priority / Risk Assessment

**High-severity bugs to fix (5 bugs that produce wrong results or crash):**
1. Notebook 4, Cell 19: `ffactor` undefined → `NameError` crash
2. Notebook 4, Cell 18: `normal(0, vn**2, Np)` → incorrect noise variance
3. Notebook 5, Cell 7: `jitter_stdev * normal(0, jitter_stdev, Np)` → squared std dev
4. Project 1, Cell 34: wrong INL plotted for 2nd sub-ADC
5. Notebook 3, Cell 15: DAC variable used in ADC error section

**Medium-risk refactoring (verify carefully):**
- Notebook 2: replacing local function definitions with imports from utils.py — verify identical behavior
- Equation corrections in notebooks 1 and 9 — verify against textbook derivations
