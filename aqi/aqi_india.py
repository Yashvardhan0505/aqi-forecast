"""India's National AQI (CPCB) for PM2.5.

The official AQI is the worst sub-index across up to eight pollutants. This project
forecasts PM2.5, which sets the AQI in Indian cities on most days, and reports its
sub-index. Breakpoints are CPCB's 24-hour PM2.5 bands; the index is capped at 500,
which PM2.5 reaches at 380 ug/m3.
"""

from __future__ import annotations

import math

# (concentration low, high in ug/m3, AQI low, high, category)
PM25_BANDS = (
    (0, 30, 0, 50, "Good"),
    (31, 60, 51, 100, "Satisfactory"),
    (61, 90, 101, 200, "Moderately polluted"),
    (91, 120, 201, 300, "Poor"),
    (121, 250, 301, 400, "Very poor"),
    (251, 380, 401, 500, "Severe"),
)
CATEGORIES = tuple(band[4] for band in PM25_BANDS)


def pm25_to_aqi(conc: float | None) -> int | None:
    """PM2.5 sub-index, linear within each CPCB band.

    Values that fall in the gap between bands (e.g. 30.5) count as the next band's
    lower edge, matching CPCB's worked example: 31 -> 51, 45 -> 75, 60 -> 100.
    """
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return None
    conc = max(0.0, float(conc))
    for c_lo, c_hi, a_lo, a_hi, _ in PM25_BANDS:
        if conc <= c_hi:
            frac = (max(conc, c_lo) - c_lo) / (c_hi - c_lo)
            return int(round(a_lo + frac * (a_hi - a_lo)))
    return 500


def category(aqi: int | None) -> str | None:
    if aqi is None:
        return None
    for _, _, a_lo, a_hi, name in PM25_BANDS:
        if aqi <= a_hi:
            return name
    return CATEGORIES[-1]


def pm25_category(conc: float | None) -> str | None:
    return category(pm25_to_aqi(conc))
