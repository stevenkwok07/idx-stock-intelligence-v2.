# IDX Stock Intelligence API V3

FastAPI backend for IDX technical screening and sector-aware fundamental screening.

## V3 changes
- Separates BANKING and NON_BANKING fundamental engines.
- Bank scoring focuses on ROE, ROA, NIM, NPL gross, CAR, LDR and CASA when the source provides them.
- Does not use conventional FCF/DCF valuation for banks.
- Non-bank scoring retains growth, profitability, leverage and cash-flow checks.
- Scores are coverage-aware: unavailable metrics are excluded from the score denominator and shown in `missing_data`; `data_coverage_percent` reports evidence coverage.
- Adds current-price fallback from historical market data when Yahoo info lacks currentPrice.
- Adds calculated ROE/ROA only when the required statement values exist.

## Deploy
Build: `pip install -r requirements.txt`
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Endpoints
- `/api/health`
- `/api/universe`
- `/api/stock/{code}`
- `/api/stock/{code}/fundamental`
- `/api/scan`
- `/api/short-term`
- `/api/trend`
- `/api/long-term`
- `/api/data-sources`
- `/docs`

## Important limitations
Yahoo Finance/yfinance is not an official IDX data feed and can be delayed, incomplete, or inconsistently labeled. V3 does not scrape undocumented IDX endpoints. Banking ratios such as NIM, NPL, CAR, LDR and CASA are not fabricated or inferred when absent from the data provider; they remain null and reduce data coverage. Confirm all figures using audited company reports and IDX disclosures. The score is a screening aid, not a forecast or investment instruction.
