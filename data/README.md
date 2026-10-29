# Data loading and checks

Install numpy, pandas and matplotlib. Run from the repository root:

```python
from data.synthetic import generate_ar1
from data.shifts import inject_shift
from data.uci import load_uci, load_uci_with_shift
from data.entsoe import load_entsoe
```

`load_uci()` returns a 1-D float array; all shift functions return
`(1-D float array, shift_at)`. Shift indices are asserted to lie in the last
20%, using the same split boundary as `chronological_split`.

`inject_shift` copies its input and estimates standard deviation using only
its first 60%. Mean severity is an offset in units of that deviation.
Variance severity is the target standard-deviation multiplier, at least 1;
independent Gaussian noise has deviation `std * sqrt(multiplier**2 - 1)`.
This targets the multiplier relative to training spread; an arbitrary input
may already have a different test-segment spread. Gradual shifts have zero
weight at onset and full weight `ramp` hours/steps later. Negative mean
severity is allowed. Defaults place onset 25% into the final 20%.

Downloads are cached in ignored `data/raw/`, independent of working directory.
Failed downloads are not published as complete cache files. First downloads
are large (UCI ZIP about 249 MB, extracted TXT about 678 MB; OPSD CSV about
124 MB). No download occurs at import time.

## Sources and conventions

- [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
  Artur Trindade (2015), CC BY 4.0, DOI 10.24432/C58C86.
  Sum 370 clients and average quarter-hour kW readings per hour, retaining
  2012-01-01 through 2014-12-31. These are hourly loads in kW, not the sum
  of four power readings. Portuguese wall-clock timestamps and UCI's DST
  zero/aggregate conventions are retained. The terminal 2015 reading is
  excluded. Parsing is chunked to reduce memory use.
- [Open Power System Data, Time series, version 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06),
  DOI 10.25832/time_series/2020-10-06; primary source ENTSO-E Transparency.
  Select `IT_load_actual_entsoe_transparency` (MW), 2019-01-01 00:00 UTC
  through 2020-05-31 23:00 UTC. Use March 9 00:00 UTC as the daily lockdown
  marker. Only Italy is supported because other countries require their own
  shift dates. Missing timestamps and values are counted via INFO logging,
  then interpolated in time; boundary gaps cause an error.
  Verified actual data: 12,408 hours, 0 missing, shift index 10,392.

## Verification

Run `python -m unittest data.test_data` for offline checks of shift arithmetic,
variance scaling, reproducibility, input preservation, invalid inputs,
semicolon/decimal-comma parsing, hourly aggregation and missing-hour filling.
Run `python -m data.plot_synthetic` to regenerate `synthetic_cases.png`, showing
mean/variance crossed with abrupt/gradual shifts and the final test segment.
To verify complete downloads, call both loaders directly.

Verified UCI download: 26,304 finite hourly total-load values.
