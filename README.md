# IDX Stock Intelligence API V4

## Deploy
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Root directory: repository root (leave blank on Render if `main.py` is at repo root).

## V4 changes
- Optional manual bank-ratio overrides sourced from published bank reports.
- Each manual override includes source, report period, and source URL in the API response.
- No sample financial values are pre-filled; missing metrics remain null.
- `/api/data-quality/{code}` shows which bank ratios are configured.
- Banking analysis does not use conventional FCF/DCF.
- Coverage-aware score remains separate from data coverage.
- No undocumented IDX API scraping is used.

## Manual bank metrics
Copy `bank_metrics.example.json` to `bank_metrics.json`, then enter values copied from an official bank report.
Values must be percentages (e.g. 3.2 means 3.2%, not 0.032). Keep missing values as `null`.
Supported fields: `nim`, `npl_gross`, `npl_net`, `car`, `ldr`, `casa`.
Set `reporting_period`, `source`, and `source_url` to the exact report used.

Example structure:
```json
{
  "BBCA": {
    "reporting_period": "2026-Q2",
    "source": "Official quarterly report",
    "source_url": "https://...",
    "metrics": {
      "nim": 5.4,
      "npl_gross": 1.8,
      "npl_net": 0.6,
      "car": 27.0,
      "ldr": 80.0,
      "casa": 82.0
    }
  }
}
```
The example numbers above are illustrative only and must not be used as BBCA data.

For Render, upload `bank_metrics.json` to the repository root or configure `BANK_METRICS_FILE` to a valid file path. Changes require a new deploy. The API does not independently verify manually entered figures.

## Important limitations
- Yahoo Finance may omit or mislabel Indonesian bank ratios.
- IDX disclosure portal is a reference for manual cross-checking, not an integrated live data API.
- Data can be delayed or incomplete.
- Scores are educational screening outputs, not forecasts or buy/sell instructions.


## V5 changes
- Strict data-quality gate: fundamental coverage below 70% always returns `INSUFFICIENT_DATA`, even if the provisional score is high.
- Adds a coverage confidence label and explicit classification-gate explanation.
- Banking score now includes NPL net as a separate criterion and rebalances bank weights to 100.
- Adds `bank_required_metrics_missing` so all six required bank ratios are explicitly reported.
- Missing values remain null; this release does not automatically fetch bank ratios from IDX.
- The score is a rule-based screening aid, not a valuation or a buy/sell instruction.

After deployment, verify `/api/health`, `/api/stock/BBCA/fundamental`, and `/api/data-quality/BBCA`.
