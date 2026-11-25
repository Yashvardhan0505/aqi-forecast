# data/

Written by the daily GitHub Action and committed, so the repo history doubles as an audit trail. Each city has its own folder (`data/jaipur/`, `data/delhi/`, ...) with these files:

| File | What's in it |
|---|---|
| `stations.csv` | The PM2.5 monitors found in the city's bounding box (OpenAQ sensor and location IDs, names, coordinates). |
| `pm25_station_daily.csv` | Daily mean PM2.5 per monitor (µg/m³), with the fraction of hours that had data. |
| `pm25_city_daily.csv` | One cleaned value per day for the whole city, and how many monitors reported. |
| `weather_daily.csv` | Daily weather rolled up from Open-Meteo's hourly data: past days, plus the forecast for the next few days. |
| `forecast_log.csv` | Every forecast ever made, appended before its day was measured and never edited: model, "same as last day" and "same as last week". |
| `backtest.csv` | Forecasts replayed for the last 180 days with only the data available at the time. Refreshed weekly. |
| `state.json` | When the station list and the backtest were last refreshed. |

Dates are calendar days in India time (IST). PM2.5 is in µg/m³.
