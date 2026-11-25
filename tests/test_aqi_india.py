import math

from aqi.aqi_india import pm25_category, pm25_to_aqi


def test_cpcb_worked_example():
    # From the CPCB / IES description of the sub-index: 31 -> 51, 45 -> 75, 60 -> 100.
    assert pm25_to_aqi(31) == 51
    assert pm25_to_aqi(45) == 75
    assert pm25_to_aqi(60) == 100


def test_band_edges_and_cap():
    assert pm25_to_aqi(0) == 0
    assert pm25_to_aqi(30) == 50
    assert pm25_to_aqi(30.5) == 51  # the gap between bands counts as the next band's floor
    assert pm25_to_aqi(90) == 200
    assert pm25_to_aqi(120) == 300
    assert pm25_to_aqi(250) == 400
    assert pm25_to_aqi(380) == 500
    assert pm25_to_aqi(900) == 500  # India's AQI is capped at 500


def test_categories():
    assert pm25_category(25) == "Good"
    assert pm25_category(55) == "Satisfactory"
    assert pm25_category(85) == "Moderately polluted"
    assert pm25_category(100) == "Poor"
    assert pm25_category(200) == "Very poor"
    assert pm25_category(300) == "Severe"


def test_missing_values():
    assert pm25_to_aqi(None) is None
    assert pm25_to_aqi(math.nan) is None
    assert pm25_category(None) is None
