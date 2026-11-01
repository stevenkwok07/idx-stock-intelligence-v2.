# IDX Stock Intelligence Backend v2

FastAPI backend for an existing frontend, with two independent rule-based engines:

1. **Short-term momentum scanner**: RSI, MACD, moving averages, momentum, abnormal volume/value traded, 20-session breakout/breakdown, ATR/volatility and liquidity flags.
2. **Long-term fundamental analyst**: revenue/profit growth, ROE, operating margin, debt/equity, operating cash flow, free cash flow, valuation multiples and an illustrative conservative DCF when inputs are sufficient.

## Important data limitations

- Market and financial data are sourced from Yahoo Finance through `yfinance`. It can be delayed, incomplete, or unavailable.
- The IDX disclosure portal is provided as an official cross-check reference. This backend deliberately does **not** scrape undocumented IDX endpoints or claim that Yahoo + IDX are already merged into a verified dataset.
- Financial statement line-item names vary. Missing values are returned as `null` and listed in `missing_data`; do not treat missing data as zero.
- The DCF is a simplified screening estimate, not a precise intrinsic value. It uses a 5-year projection, 12% discount rate and 3% terminal growth, with growth capped at 10%. Banks and financial companies need sector-specific valuation models; do not rely on this generic FCF DCF for them.
- ROE and operating margin are only moat proxies. An economic moat requires qualitative research into competitive advantage, industry structure, pricing power, switching costs and capital allocation.
- Short-term candidates are watchlist signals, not predictions of manipulation, pump activity, or guaranteed bullish returns.

## API endpoints

- `GET /api/health`
- `GET /api/universe`
- `GET /api/stock/BBCA` — technical analysis; keeps the legacy endpoint shape and core fields.
- `GET /api/stock/BBCA/fundamental` — fundamental analysis for one company.
- `GET /api/scan?limit=30` — ranked technical scan.
- `GET /api/short-term?limit=30` — bullish technical watchlist candidates.
- `GET /api/trend` — bullish / bearish / sideways groups.
- `GET /api/long-term?limit=30` — ranked fundamental screen.
- `GET /api/data-sources` — provider and limitation notes.
- `POST /api/cache/clear`
- `GET /docs`

## Configuration

Environment variables:

- `CORS_ORIGINS=*` for development; set this to your exact Netlify origin in production, e.g. `https://your-site.netlify.app`.
- `TICKERS=BBCA.JK,BBRI.JK,TLKM.JK` to replace the starter universe. Supply a comma-separated list of Yahoo Finance ticker symbols.
- `CACHE_TTL_SECONDS=900` for price cache duration.
- `HISTORY_PERIOD=1y` for technical history.
- `MIN_AVG_VALUE_IDR=1000000000` for minimum average daily traded value over the prior 20 sessions.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Render deployment

Use the included `render.yaml`. Build command: `pip install -r requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

For public production use, configure `CORS_ORIGINS` to the exact frontend origin and consider a background job/provider with a documented market-data license and rate limits. The synchronous scan endpoints can take time because they request each ticker sequentially.

## Frontend compatibility

Existing `/api/scan`, `/api/trend`, and `/api/stock/{code}` endpoints remain. New dashboard tabs can call:

```js
const [shortTerm, longTerm] = await Promise.all([
  fetch(`${API_BASE}/api/short-term?limit=30`).then(r => r.json()),
  fetch(`${API_BASE}/api/long-term?limit=30`).then(r => r.json()),
]);
```

The frontend's existing technical endpoint remains available. Fundamental endpoints are additive.
