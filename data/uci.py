"""UCI ElectricityLoadDiagrams20112014: hourly total load in kW."""
import shutil
import zipfile
import numpy as np
import pandas as pd
from ._download import RAW_DIR, download
from .shifts import inject_shift

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"


def load_uci():
    """Return hourly mean total kW from 2012-01-01 through 2014-12-31.

    Cache raw files under data/raw. Read in chunks to avoid loading the
    entire 370-client table into memory. Timestamps retain UCI's Portuguese
    wall-clock convention, including its documented DST adjustments.
    """
    path = RAW_DIR / "LD2011_2014.txt"
    if not path.exists():
        archive = download(URL, "electricityloaddiagrams20112014.zip")
        temporary = path.with_suffix(".txt.part")
        try:
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open("LD2011_2014.txt") as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    totals = []
    for chunk in pd.read_csv(path, sep=";", decimal=",", index_col=0,
                             parse_dates=True, chunksize=10000):
        totals.append(chunk.sum(axis=1, min_count=chunk.shape[1]))
    hourly = pd.concat(totals).sort_index().resample("h").mean()
    hourly = hourly.loc["2012-01-01":"2014-12-31"]
    values = hourly.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("UCI load contains missing or non-finite hourly values")
    return values


def load_uci_with_shift(shift_type, speed, severity, seed=0):
    """Return (hourly load with an injected test-segment shift, shift_at)."""
    return inject_shift(load_uci(), shift_type, speed, severity, seed=seed)
