import os
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("idx-stock-intelligence")

DEFAULT_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "GOTO.JK", "BYAN.JK", "UNVR.JK", "HMSP.JK", "ICBP.JK",
    "KLBF.JK", "SIDO.JK", "EXCL.JK", "ISAT.JK", "MNCN.JK",
    "ADRO.JK", "PTBA.JK", "INCO.JK", "ANTM.JK", "MDKA.JK",
    "SMGR.JK", "INKP.JK", "TKIM.JK", "JPFA.JK", "CPIN.JK",
    "ACES.JK", "MAPI.JK", "LPPF.JK", "ERAA.JK", "HEAL.JK",
]
# Optional expansion without code changes, e.g. TICKERS=BBCA.JK,BBRI.JK,TLKM.JK
_env_tickers = [x.strip().upper() for x in os.getenv("TICKERS", "").split(",") if x.strip()]
TICKERS = list(dict.fromkeys(_env_tickers or DEFAULT_TICKERS))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "1y")
MIN_AVG_VALUE_IDR = float(os.getenv("MIN_AVG_VALUE_IDR", "1000000000"))
# IDX public disclosure portal is exposed as a source/reference, not scraped as an undocumented API.
IDX_DISCLOSURE_URL = "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/"

app = FastAPI(
    title="IDX Stock Intelligence API",
    version="3.0.0",
    description="IDX screener with technical momentum and sector-specific, coverage-aware fundamental analysis.",
)

origins_env = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if origins_env.strip() == "*" else [x.strip() for x in origins_env.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Small in-process TTL cache. This avoids stale forever-caches from lru_cache.
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def cache_get_or_create(key: str, ttl: int, factory):
    now = time.time()
    with _cache_lock:
        item = _cache.get(key)
        if item and now - item[0] < ttl:
            return item[1]
    value = factory()
    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value


def clear_all_cache():
    with _cache_lock:
        _cache.clear()


def clean_num(value, digits=4):
    try:
        value = float(value)
        return round(value, digits) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def normalize_code(code: str) -> str:
    ticker = code.upper().strip()
    if not ticker.endswith(".JK"):
        ticker += ".JK"
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Kode saham tidak ada di universe konfigurasi.")
    return ticker


def fetch_history(ticker: str) -> pd.DataFrame:
    def load():
        df = yf.Ticker(ticker).history(period=HISTORY_PERIOD, auto_adjust=True, actions=False)
        if df is None or df.empty:
            raise ValueError(f"Data historis tidak tersedia untuk {ticker}")
        df = df.dropna(subset=["Close", "High", "Low"])
        if len(df) < 55:
            raise ValueError(f"Data historis kurang dari 55 sesi untuk {ticker}")
        return df
    return cache_get_or_create(f"history:{ticker}:{HISTORY_PERIOD}", CACHE_TTL_SECONDS, load)


def _rsi(close: pd.Series, period=14) -> float:
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    value = safe_float(result.iloc[-1])
    if value is None:
        return 100.0 if safe_float(avg_loss.iloc[-1]) == 0 else 50.0
    return value


def analyze_technical(ticker: str) -> dict:
    df = fetch_history(ticker)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].fillna(0).astype(float)
    value_traded = close * volume
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    ma = {n: float(close.rolling(n).mean().iloc[-1]) for n in (5, 10, 20, 50)}
    rsi = _rsi(close)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - macd_signal

    vol_avg20 = float(volume.iloc[-21:-1].mean())
    vol_ratio = float(volume.iloc[-1] / vol_avg20) if vol_avg20 > 0 else 0.0
    value_avg20 = float(value_traded.iloc[-21:-1].mean())
    value_ratio = float(value_traded.iloc[-1] / value_avg20) if value_avg20 > 0 else 0.0
    # Exclude today's bar from the breakout reference to avoid look-ahead leakage.
    prior_high20 = float(high.iloc[-21:-1].max())
    prior_low20 = float(low.iloc[-21:-1].min())
    breakout = price > prior_high20
    breakdown = price < prior_low20

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / price * 100) if price else None
    returns = close.pct_change().dropna()
    volatility20 = float(returns.tail(20).std() * np.sqrt(252) * 100) if len(returns) >= 20 else None
    momentum5 = (price / float(close.iloc[-6]) - 1) * 100
    momentum20 = (price / float(close.iloc[-21]) - 1) * 100
    avg_value20 = value_avg20

    score = 0
    max_score = 100
    positives, negatives, risks = [], [], []
    checks = [
        (price > ma[20], 12, "Harga di atas MA20"),
        (ma[20] > ma[50], 10, "MA20 di atas MA50"),
        (float(macd.iloc[-1]) > float(macd_signal.iloc[-1]), 10, "MACD di atas signal"),
        (float(hist.iloc[-1]) > float(hist.iloc[-2]), 6, "Histogram MACD membaik"),
        (40 <= rsi <= 68, 8, "RSI berada di zona momentum moderat"),
        (momentum5 > 0, 6, "Momentum 5 sesi positif"),
        (momentum20 > 0, 8, "Momentum 20 sesi positif"),
        (vol_ratio >= 1.5, 10, "Volume minimal 1,5x rata-rata 20 sesi"),
        (breakout, 15, "Harga menembus high 20 sesi sebelumnya"),
        (avg_value20 >= MIN_AVG_VALUE_IDR, 15, "Nilai transaksi rata-rata memenuhi ambang likuiditas"),
    ]
    for condition, points, label in checks:
        if condition:
            score += points
            positives.append(label)
    if rsi > 75:
        risks.append("RSI sangat tinggi: risiko kondisi jenuh beli/kenaikan terlalu cepat")
    if atr_pct is not None and atr_pct > 7:
        risks.append("ATR% tinggi: rentang pergerakan harian relatif lebar")
    if avg_value20 < MIN_AVG_VALUE_IDR:
        risks.append("Likuiditas di bawah ambang screener; harga dapat sulit dieksekusi")
    if value_ratio >= 3 and abs(momentum5) < 1:
        risks.append("Nilai transaksi melonjak tanpa kemajuan harga yang sebanding; perlu pemeriksaan manual")
    if breakdown:
        negatives.append("Harga menembus low 20 sesi sebelumnya")
    if price < ma[20]:
        negatives.append("Harga di bawah MA20")
    if vol_ratio < 0.7:
        negatives.append("Volume lebih rendah dari kebiasaan")

    # A watchlist signal is not a buy instruction and requires independent verification.
    signal = "WATCHLIST_BULLISH" if score >= 65 and not breakdown else "NO_CLEAR_SIGNAL"
    if breakdown or (price < ma[20] and float(macd.iloc[-1]) < float(macd_signal.iloc[-1])):
        signal = "WEAK_TREND"
    trend = "BULLISH" if price > ma[20] and float(macd.iloc[-1]) > float(macd_signal.iloc[-1]) else (
        "BEARISH" if price < ma[20] and float(macd.iloc[-1]) < float(macd_signal.iloc[-1]) else "SIDEWAYS"
    )
    hist_rows = []
    for date, row in df.tail(60).iterrows():
        hist_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": clean_num(row["Close"]),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
        })
    return {
        "ticker": ticker, "code": ticker.replace(".JK", ""), "name": ticker.replace(".JK", ""),
        "currency": "IDR", "price": clean_num(price), "previous_close": clean_num(prev),
        "change_percent": clean_num((price / prev - 1) * 100 if prev else None),
        "rsi": clean_num(rsi), "macd": clean_num(macd.iloc[-1]),
        "macd_signal": clean_num(macd_signal.iloc[-1]), "macd_histogram": clean_num(hist.iloc[-1]),
        "ma5": clean_num(ma[5]), "ma10": clean_num(ma[10]), "ma20": clean_num(ma[20]), "ma50": clean_num(ma[50]),
        "support": clean_num(prior_low20), "resistance": clean_num(prior_high20),
        "momentum_5d_percent": clean_num(momentum5), "momentum_20d_percent": clean_num(momentum20),
        "volume_ratio_20d": clean_num(vol_ratio), "value_traded_ratio_20d": clean_num(value_ratio),
        "average_value_traded_20d_idr": clean_num(avg_value20, 0),
        "atr_percent": clean_num(atr_pct), "annualized_volatility_20d_percent": clean_num(volatility20),
        "breakout_20d": bool(breakout), "breakdown_20d": bool(breakdown),
        "score": score, "score_percent": score, "max_score": max_score,
        "signal": signal, "trend": trend,
        "summary": "Masuk radar untuk pemeriksaan lanjutan" if signal == "WATCHLIST_BULLISH" else "Belum ada kombinasi sinyal yang cukup kuat",
        "details": positives + negatives, "risk_flags": risks,
        "history": hist_rows, "data_as_of": df.index[-1].strftime("%Y-%m-%d"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance; historical market data may be delayed or incomplete",
    }


def _statement_value(statement: pd.DataFrame, aliases: list[str], period_col=None):
    if statement is None or statement.empty:
        return None
    for alias in aliases:
        matches = [idx for idx in statement.index if str(idx).lower() == alias.lower()]
        if matches:
            row = statement.loc[matches[0]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if period_col is not None and period_col in row.index:
                return safe_float(row[period_col])
            for col in row.index:
                val = safe_float(row[col])
                if val is not None:
                    return val
    return None


def _get_series(statement: pd.DataFrame, aliases: list[str]) -> list[tuple[str, float]]:
    if statement is None or statement.empty:
        return []
    for alias in aliases:
        matches = [idx for idx in statement.index if str(idx).lower() == alias.lower()]
        if matches:
            row = statement.loc[matches[0]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            values = []
            for col, val in row.items():
                num = safe_float(val)
                if num is not None:
                    try:
                        year = pd.Timestamp(col).strftime("%Y-%m-%d")
                    except Exception:
                        year = str(col)
                    values.append((year, num))
            return sorted(values, reverse=True)
    return []


def fetch_fundamentals(ticker: str) -> dict:
    def load():
        obj = yf.Ticker(ticker)
        info = obj.get_info() or {}
        income = obj.financials
        cashflow = obj.cashflow
        balance = obj.balance_sheet
        return {"info": info, "income": income, "cashflow": cashflow, "balance": balance}
    return cache_get_or_create(f"fundamentals:{ticker}", max(CACHE_TTL_SECONDS, 3600), load)



BANK_TICKERS = {
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BNGA.JK",
    "BDMN.JK", "NISP.JK", "MEGA.JK", "ARTO.JK", "BBTN.JK", "BTPN.JK",
    "BTPS.JK", "BACA.JK", "AGRO.JK", "BJBR.JK", "BJTM.JK",
}

def _first_info(info: dict, keys: list[str]):
    for key in keys:
        value = safe_float(info.get(key))
        if value is not None:
            return value
    return None

def _ratio_percent(value):
    value = safe_float(value)
    if value is None:
        return None
    # Yahoo ratios are usually decimals, while some reported data may already be percent.
    return value * 100 if abs(value) <= 1.5 else value

def _latest_and_previous(series):
    return (series[0][1], series[1][1]) if len(series) >= 2 else (
        series[0][1] if series else None, None
    )

def analyze_fundamental(ticker: str) -> dict:
    raw = fetch_fundamentals(ticker)
    info, income, cashflow, balance = raw["info"], raw["income"], raw["cashflow"], raw["balance"]
    is_bank = ticker in BANK_TICKERS

    revenue_s = _get_series(income, ["Total Revenue", "Operating Revenue", "Total Non Interest Income"])
    profit_s = _get_series(income, ["Net Income", "Net Income Common Stockholders"])
    cfo_s = _get_series(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex_s = _get_series(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    fcf_s = _get_series(cashflow, ["Free Cash Flow"])

    latest_income_col = income.columns[0] if income is not None and not income.empty else None
    latest_balance_col = balance.columns[0] if balance is not None and not balance.empty else None
    previous_balance_col = balance.columns[1] if balance is not None and len(balance.columns) > 1 else None

    revenue = _statement_value(income, ["Total Revenue", "Operating Revenue", "Total Non Interest Income"], latest_income_col)
    net_income = _statement_value(income, ["Net Income", "Net Income Common Stockholders"], latest_income_col)
    pretax_income = _statement_value(income, ["Pretax Income", "Pretax Income (Loss)", "Income Before Tax"], latest_income_col)
    operating_income = _statement_value(income, ["Operating Income"], latest_income_col)
    total_debt = _statement_value(balance, ["Total Debt"], latest_balance_col)
    total_equity = _statement_value(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"], latest_balance_col)
    previous_equity = _statement_value(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"], previous_balance_col)
    total_assets = _statement_value(balance, ["Total Assets"], latest_balance_col)
    previous_assets = _statement_value(balance, ["Total Assets"], previous_balance_col)
    cash = _statement_value(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], latest_balance_col)
    cfo = _statement_value(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = _statement_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    fcf = _statement_value(cashflow, ["Free Cash Flow"])
    if fcf is None and cfo is not None and capex is not None:
        fcf = cfo + capex if capex < 0 else cfo - capex

    revenue_latest, revenue_previous = _latest_and_previous(revenue_s)
    profit_latest, profit_previous = _latest_and_previous(profit_s)
    revenue_growth = ((revenue_latest / revenue_previous) - 1) * 100 if revenue_latest is not None and revenue_previous not in (None, 0) else None
    profit_growth = ((profit_latest / profit_previous) - 1) * 100 if profit_latest is not None and profit_previous not in (None, 0) else None

    shares = _first_info(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    eps = _first_info(info, ["trailingEps", "forwardEps"])
    book_value = _first_info(info, ["bookValue"])
    current_price = _first_info(info, ["currentPrice", "regularMarketPrice"])
    if current_price is None:
        try:
            hist = fetch_history(ticker)
            current_price = safe_float(hist["Close"].iloc[-1])
        except Exception:
            pass
    market_cap = _first_info(info, ["marketCap"])
    if market_cap is None and current_price is not None and shares is not None:
        market_cap = current_price * shares

    # Prefer reported ratios; derive only ratios whose numerator/denominator are clearly identified.
    roe = _ratio_percent(_first_info(info, ["returnOnEquity"]))
    if roe is None and net_income is not None and total_equity and previous_equity:
        avg_equity = (total_equity + previous_equity) / 2
        roe = net_income / avg_equity * 100 if avg_equity else None

    roa = _ratio_percent(_first_info(info, ["returnOnAssets"]))
    if roa is None and pretax_income is not None and total_assets and previous_assets:
        avg_assets = (total_assets + previous_assets) / 2
        roa = pretax_income / avg_assets * 100 if avg_assets else None

    nim = _ratio_percent(_first_info(info, ["netInterestMargin", "netInterestMarginPercent", "nim"]))
    npl_gross = _ratio_percent(_first_info(info, ["nonPerformingLoan", "nonPerformingLoans", "nplGross", "nplGrossPercent"]))
    npl_net = _ratio_percent(_first_info(info, ["nplNet", "nplNetPercent"]))
    car = _ratio_percent(_first_info(info, ["capitalAdequacyRatio", "capitalAdequacy", "car"]))
    ldr = _ratio_percent(_first_info(info, ["loanToDepositRatio", "loanToDeposit", "ldr"]))
    casa = _ratio_percent(_first_info(info, ["casaRatio", "casa"]))

    operating_margin = _ratio_percent(_first_info(info, ["operatingMargins"]))
    debt_equity = _first_info(info, ["debtToEquity"])
    if debt_equity is not None and abs(debt_equity) > 10:
        debt_equity = debt_equity / 100
    if debt_equity is None and total_debt is not None and total_equity and total_equity > 0:
        debt_equity = total_debt / total_equity

    price_to_book = _first_info(info, ["priceToBook"])
    if price_to_book is None and market_cap is not None and total_equity and total_equity > 0:
        price_to_book = market_cap / total_equity
    trailing_pe = _first_info(info, ["trailingPE"])
    if trailing_pe is None and current_price and eps and eps > 0:
        trailing_pe = current_price / eps

    # V3 uses a coverage-aware score: missing data is excluded from the denominator,
    # and a separate coverage percentage tells users how much evidence is available.
    positives, risks, missing = [], [], []
    criteria = []
    if is_bank:
        criteria = [
            (revenue_growth, lambda x: x > 5, 10, "Pertumbuhan pendapatan >5% YoY"),
            (profit_growth, lambda x: x > 5, 10, "Pertumbuhan laba bersih >5% YoY"),
            (roe, lambda x: x >= 12, 15, "ROE minimal 12%"),
            (roa, lambda x: x >= 1.5, 10, "ROA minimal 1,5%"),
            (nim, lambda x: x >= 3.5, 15, "NIM minimal 3,5%"),
            (npl_gross, lambda x: x <= 3, 15, "NPL gross maksimal 3%"),
            (car, lambda x: x >=  12, 15, "CAR memenuhi batas evaluasi internal"),
            (ldr, lambda x: 75 <= x <= 92, 5, "LDR berada pada rentang evaluasi internal"),
            (casa, lambda x: x >= 50, 5, "CASA minimal 50%"),
        ]
    else:
        criteria = [
            (revenue_growth, lambda x: x > 5, 12, "Pendapatan tumbuh >5% YoY"),
            (profit_growth, lambda x: x > 5, 12, "Laba bersih tumbuh >5% YoY"),
            (roe, lambda x: x >= 15, 12, "ROE minimal 15%"),
            (operating_margin, lambda x: x >= 15, 10, "Operating margin minimal 15%"),
            (debt_equity, lambda x: x <= 1, 10, "Debt/equity maksimal 1x"),
            (cfo, lambda x: x > 0, 12, "Arus kas operasi positif"),
            (fcf, lambda x: x > 0, 12, "Free cash flow positif"),
            (total_equity, lambda x: x > 0, 8, "Ekuitas positif"),
            (net_income, lambda x: x > 0, 6, "Laba bersih positif"),
            (cash if cash is not None and total_debt is not None else None,
             lambda x: x >= total_debt * 0.2, 6, "Kas material dibandingkan utang"),
        ]

    available_weight = 0
    earned_weight = 0
    for value, test, weight, label in criteria:
        if value is None:
            missing.append(label.replace("minimal", "").replace("maksimal", "") + " — data tidak tersedia")
            continue
        available_weight += weight
        if test(value):
            earned_weight += weight
            positives.append(label)

    score = round(earned_weight / available_weight * 100) if available_weight else 0
    total_weight = sum(x[2] for x in criteria)
    coverage = round(available_weight / total_weight * 100) if total_weight else 0

    if revenue_growth is not None and revenue_growth < 0:
        risks.append("Pendapatan mengalami kontraksi")
    if profit_growth is not None and profit_growth < 0:
        risks.append("Laba bersih mengalami penurunan")
    if net_income is not None and net_income < 0:
        risks.append("Perusahaan mencatat rugi bersih")
    if is_bank:
        if npl_gross is not None and npl_gross > 5:
            risks.append("NPL gross di atas 5%; perlu pemeriksaan kualitas kredit")
        if car is not None and car < 12:
            risks.append("CAR di bawah ambang evaluasi internal 12%; verifikasi ketentuan dan data terbaru")
        if ldr is not None and (ldr > 100 or ldr < 60):
            risks.append("LDR berada di luar rentang pemantauan internal")
    else:
        if cfo is not None and cfo < 0:
            risks.append("Arus kas operasi negatif")
        if fcf is not None and fcf < 0:
            risks.append("Free cash flow negatif")
        if debt_equity is not None and debt_equity > 2:
            risks.append("Debt/equity lebih dari 2x")

    # Conventional FCF-DCF is deliberately disabled for banks.
    intrinsic_value = None
    margin_of_safety = None
    dcf_assumptions = None
    if not is_bank and fcf is not None and fcf > 0 and shares and shares > 0 and current_price:
        growth = max(0.0, min((revenue_growth or 0.0) / 100, 0.08))
        discount, terminal = 0.12, 0.03
        projected = [fcf * ((1 + growth) ** year) for year in range(1, 6)]
        pv = sum(value / ((1 + discount) ** year) for year, value in enumerate(projected, 1))
        terminal_value = projected[-1] * (1 + terminal) / (discount - terminal)
        intrinsic_value = (pv + terminal_value / ((1 + discount) ** 5)) / shares
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value * 100 if intrinsic_value > 0 else None
        dcf_assumptions = {"projection_years": 5, "discount_rate_percent": 12, "terminal_growth_percent": 3,
                           "growth_cap_percent": 8, "warning": "Illustrative only; validate cash flows and assumptions manually."}

    if is_bank:
        classification = "INSUFFICIENT_DATA" if coverage < 40 else (
            "QUALITY_CANDIDATE" if score >= 75 and coverage >= 70 and not risks else (
                "REVIEW" if score >= 55 else "WEAK_OR_UNPROVEN"))
        moat_proxy = "Not assessed automatically"
    else:
        classification = "INSUFFICIENT_DATA" if coverage < 40 else (
            "QUALITY_CANDIDATE" if score >= 75 and coverage >= 70 and not risks else (
                "REVIEW" if score >= 55 else "WEAK_OR_UNPROVEN"))
        moat_proxy = "Potential profitability strength; not proof of moat" if (roe or 0) >= 15 and (operating_margin or 0) >= 12 else "Insufficient evidence"

    return {
        "ticker": ticker, "code": ticker.replace(".JK", ""),
        "name": info.get("longName") or info.get("shortName") or ticker.replace(".JK", ""),
        "sector_engine": "BANKING" if is_bank else "NON_BANKING",
        "currency": info.get("currency", "IDR"),
        "current_price": clean_num(current_price), "market_cap": clean_num(market_cap, 0),
        "revenue_latest": clean_num(revenue, 0), "net_income_latest": clean_num(net_income, 0),
        "operating_income_latest": clean_num(operating_income, 0),
        "revenue_growth_yoy_percent": clean_num(revenue_growth),
        "net_income_growth_yoy_percent": clean_num(profit_growth),
        "roe_percent": clean_num(roe), "roa_percent": clean_num(roa),
        "nim_percent": clean_num(nim), "npl_gross_percent": clean_num(npl_gross),
        "npl_net_percent": clean_num(npl_net), "car_percent": clean_num(car),
        "ldr_percent": clean_num(ldr), "casa_percent": clean_num(casa),
        "operating_margin_percent": clean_num(operating_margin),
        "debt_to_equity": clean_num(debt_equity), "total_debt": clean_num(total_debt, 0),
        "total_equity": clean_num(total_equity, 0), "cash_and_equivalents": clean_num(cash, 0),
        "operating_cash_flow": clean_num(cfo, 0),
        "capital_expenditure": clean_num(capex, 0),
        "free_cash_flow": clean_num(fcf, 0) if not is_bank else None,
        "trailing_eps": clean_num(eps), "book_value_per_share": clean_num(book_value),
        "trailing_pe": clean_num(trailing_pe), "price_to_book": clean_num(price_to_book),
        "intrinsic_value_per_share_estimate": clean_num(intrinsic_value),
        "margin_of_safety_percent": clean_num(margin_of_safety),
        "dcf_assumptions": dcf_assumptions,
        "moat_proxy": moat_proxy,
        "moat_warning": "Moat tidak dapat dipastikan dari rasio otomatis; perlu riset pangsa pasar, switching cost, brand, pricing power, dan daya tahan keuntungan.",
        "score": score, "score_percent": score, "max_score": 100,
        "data_coverage_percent": coverage, "available_weight": available_weight,
        "classification": classification, "positives": positives,
        "risk_flags": risks, "missing_data": missing,
        "financial_periods_available": {
            "revenue": [x[0] for x in revenue_s], "net_income": [x[0] for x in profit_s],
            "operating_cash_flow": [x[0] for x in cfo_s], "free_cash_flow": [x[0] for x in fcf_s],
        },
        "scoring_method": "Coverage-aware sector-specific rule scoring; missing metrics are excluded from score denominator and reported separately.",
        "data_sources": [
            {"name": "Yahoo Finance/yfinance", "type": "market_and_financial_data"},
            {"name": "IDX disclosure portal", "url": IDX_DISCLOSURE_URL, "type": "official_cross_check_reference"},
        ],
        "data_note": "Yahoo Finance coverage and labels may be incomplete. Bank-specific ratios are only scored when supplied; missing NIM/NPL/CAR/LDR/CASA are not inferred. Verify against audited reports and IDX disclosures.",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    return {
        "name": "IDX Stock Intelligence API", "version": "3.0.0", "status": "ok", "docs": "/docs",
        "endpoints": [
            "/api/health", "/api/universe", "/api/scan", "/api/trend",
            "/api/short-term", "/api/long-term", "/api/stock/BBCA",
            "/api/stock/BBCA/fundamental", "/api/cache/clear",
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "idx-stock-intelligence", "version": "3.0.0"}


@app.get("/api/universe")
def universe():
    return {"count": len(TICKERS), "tickers": TICKERS, "configuration": "TICKERS environment variable or built-in starter universe"}


@app.get("/api/stock/{code}")
def stock(code: str):
    ticker = normalize_code(code)
    try:
        return analyze_technical(ticker)
    except Exception as exc:
        log.exception("Technical analysis failed for %s", ticker)
        raise HTTPException(status_code=502, detail=f"Data teknikal tidak tersedia untuk {ticker}: {str(exc)}") from exc


@app.get("/api/stock/{code}/fundamental")
def stock_fundamental(code: str):
    ticker = normalize_code(code)
    try:
        return analyze_fundamental(ticker)
    except Exception as exc:
        log.exception("Fundamental analysis failed for %s", ticker)
        raise HTTPException(status_code=502, detail=f"Data fundamental tidak tersedia untuk {ticker}: {str(exc)}") from exc


@app.get("/api/scan")
def scan(limit: int = Query(default=30, ge=1, le=200)):
    results, errors = [], []
    for ticker in TICKERS:
        try:
            results.append(analyze_technical(ticker))
        except Exception as exc:
            log.warning("Skip technical %s: %s", ticker, exc)
            errors.append({"ticker": ticker, "reason": str(exc)})
    results.sort(key=lambda x: (x["score"], x["momentum_5d_percent"] or -999), reverse=True)
    return {"count": len(results[:limit]), "unavailable": errors, "results": results[:limit],
            "data_as_of": max((x["data_as_of"] for x in results), default=None),
            "note": "Historical data may be delayed. A scanner score is not a forecast or buy instruction."}


@app.get("/api/short-term")
def short_term(limit: int = Query(default=30, ge=1, le=200)):
    payload = scan(limit)
    candidates = [x for x in payload["results"] if x["signal"] == "WATCHLIST_BULLISH"]
    return {
        "engine": "short_term_momentum", "count": len(candidates), "candidates": candidates,
        "unavailable": payload["unavailable"],
        "method": "Technical rules: trend, momentum, RSI, MACD, volume/value anomaly, 20-session breakout and liquidity filter.",
        "warning": "High-risk watchlist only. It does not predict a pump, establish manipulation, or guarantee bullish movement.",
        "data_as_of": payload["data_as_of"],
    }


@app.get("/api/trend")
def trend():
    groups = {"bullish": [], "bearish": [], "sideways": []}
    unavailable = []
    for ticker in TICKERS:
        try:
            item = analyze_technical(ticker)
            groups[item["trend"].lower()].append(item)
        except Exception as exc:
            log.warning("Skip %s: %s", ticker, exc)
            unavailable.append({"ticker": ticker, "reason": str(exc)})
    groups["bullish"].sort(key=lambda x: x["score"], reverse=True)
    groups["bearish"].sort(key=lambda x: x["score"])
    return {**{k: v[:10] for k, v in groups.items()}, "unavailable": unavailable}


@app.get("/api/long-term")
def long_term(limit: int = Query(default=30, ge=1, le=200)):
    results, errors = [], []
    for ticker in TICKERS:
        try:
            results.append(analyze_fundamental(ticker))
        except Exception as exc:
            log.warning("Skip fundamental %s: %s", ticker, exc)
            errors.append({"ticker": ticker, "reason": str(exc)})
    results.sort(key=lambda x: (
        x["classification"] == "QUALITY_CANDIDATE",
        x["score"],
        x["margin_of_safety_percent"] if x["margin_of_safety_percent"] is not None else -999,
    ), reverse=True)
    return {
        "engine": "long_term_fundamental", "count": len(results[:limit]),
        "results": results[:limit], "unavailable": errors,
        "method": "Financial statement rules and conservative illustrative DCF where inputs are available.",
        "warning": "Not investment advice. Yahoo statement coverage can be incomplete; verify audited statements and qualitative moat evidence.",
        "idx_disclosure_url": IDX_DISCLOSURE_URL,
    }


@app.post("/api/cache/clear")
def clear_cache_endpoint():
    clear_all_cache()
    return {"status": "cleared"}


@app.get("/api/data-sources")
def data_sources():
    return {
        "market_data": {"provider": "Yahoo Finance via yfinance", "limitations": ["May be delayed", "Coverage can change", "Not guaranteed real-time"]},
        "official_cross_check": {"provider": "Indonesia Stock Exchange (IDX)", "url": IDX_DISCLOSURE_URL, "usage": "Manual/official verification reference; this backend does not call an undocumented IDX API."},
        "fundamental_engines": {"banking": ["ROE", "ROA", "NIM", "NPL", "CAR", "LDR", "CASA"], "non_banking": ["growth", "ROE", "operating_margin", "debt_equity", "CFO", "FCF", "illustrative_DCF"]},
    }
