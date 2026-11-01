"""Italy load from OPSD 2020-10-06, with the real COVID lockdown shift."""
import logging
import numpy as np
import pandas as pd
from ._download import download

URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
logger = logging.getLogger(__name__)


def load_entsoe(country="IT"):
    """Return (hourly load in MW, lockdown index), on a UTC time axis.

    Includes all hours from 2019-01-01 through 2020-05-31. Missing timestamps
    and values are counted and logged before time interpolation. Missing
    boundary values raise an error rather than silently extrapolating.
    Only Italy has a defined lockdown date in this experiment.
    """
    if country.upper() != "IT":
        raise ValueError("Only IT is supported: its shift date is 2020-03-09")
    column = "IT_load_actual_entsoe_transparency"
    path = download(URL, "time_series_60min_singleindex.csv")
    frame = pd.read_csv(path, usecols=["utc_timestamp", column])
    timestamps = pd.to_datetime(frame["utc_timestamp"], utc=True)
    load = pd.Series(frame[column].to_numpy(dtype=float), index=timestamps).sort_index()
    if load.index.has_duplicates:
        raise ValueError("ENTSO-E contains duplicate UTC timestamps")
    hours = pd.date_range("2019-01-01", "2020-05-31 23:00", freq="h", tz="UTC")
    load = load.reindex(hours)
    missing = int(load.isna().sum())
    logger.info("ENTSO-E IT: interpolating %d missing hours out of %d", missing, len(load))
    load = load.interpolate(method="time", limit_area="inside")
    values = load.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ENTSO-E has unfillable boundary gaps or non-finite load")
    shift_at = int(hours.get_loc(pd.Timestamp("2020-03-09", tz="UTC")))
    assert int(0.8 * len(values)) <= shift_at < len(values), "shift must be in last 20%"
    return values, shift_at
