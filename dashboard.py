import base64
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Trade Decision Cockpit",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
.trade-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 18px;
    padding: 1.0rem 1.1rem;
    background: rgba(128,128,128,0.06);
    margin-bottom: 0.75rem;
}
.big-status {
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: .35rem;
}
.trade-muted {opacity: .75; font-size: .92rem;}
.good {color: #16a34a;}
.warn {color: #d97706;}
.bad {color: #dc2626;}
.neutral {color: #64748b;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Secrets / GitHub state
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def github_config() -> dict:
    return {
        "token": get_secret("GITHUB_STATE_TOKEN"),
        "owner": get_secret("GITHUB_STATE_OWNER"),
        "repo": get_secret("GITHUB_STATE_REPO"),
        "branch": get_secret("GITHUB_STATE_BRANCH", "main"),
        "file": get_secret("GITHUB_STATE_FILE", "trade_manager_open_trades.json"),
    }


@st.cache_data(ttl=60)
def load_trade_state_from_github() -> dict:
    cfg = github_config()
    missing = [k for k, v in cfg.items() if not v and k != "branch"]
    if missing:
        return {
            "error": f"Missing Streamlit secret(s): {', '.join(missing)}",
            "open_trades": [],
            "closed_trades": [],
            "latest_scan": [],
        }

    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['file']}?ref={cfg['branch']}"
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if not resp.ok:
            return {
                "error": f"GitHub API error {resp.status_code}: {resp.text[:300]}",
                "open_trades": [],
                "closed_trades": [],
                "latest_scan": [],
            }

        payload = resp.json()
        decoded = base64.b64decode(payload.get("content", "")).decode("utf-8")
        data = json.loads(decoded)
        data.setdefault("open_trades", [])
        data.setdefault("closed_trades", [])
        data.setdefault("latest_scan", [])
        data.setdefault("last_updated", "")
        return data

    except Exception as exc:
        return {
            "error": f"Could not load GitHub state file: {exc}",
            "open_trades": [],
            "closed_trades": [],
            "latest_scan": [],
        }


# ============================================================
# Yahoo Finance data
# ============================================================

YAHOO_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",

    # Gold futures proxy for Yahoo Finance
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",

    # Index futures proxies for Yahoo Finance
    "US100": "NQ=F",
    "NAS100": "NQ=F",
    "US500": "ES=F",
    "SPX500": "ES=F",
}


def normalize_yahoo_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace(" ", "").replace("/", "")
    return YAHOO_MAP.get(s, symbol.strip().upper())


def effective_period_for_interval(period: str, interval: str) -> str:
    intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if interval in intraday and period in {"1y", "2y", "5y"}:
        return "60d"
    if interval == "1m" and period not in {"1d", "5d", "7d"}:
        return "7d"
    return period


@st.cache_data(ttl=120)
def load_yahoo_data(symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    yf_symbol = normalize_yahoo_symbol(symbol)
    try:
        df = yf.download(
            yf_symbol,
            period=effective_period_for_interval(period, interval),
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )

        if df.empty:
            return pd.DataFrame(), f"No Yahoo Finance data returned for {yf_symbol}."

        if isinstance(df.columns, pd.MultiIndex):
            picked_level = None
            required = {"Open", "High", "Low", "Close"}
            for level in range(df.columns.nlevels):
                values = set(map(str, df.columns.get_level_values(level)))
                if required.issubset(values):
                    picked_level = level
                    break
            if picked_level is not None:
                df.columns = df.columns.get_level_values(picked_level)
            else:
                df.columns = [str(c[-1] if isinstance(c, tuple) else c) for c in df.columns]

        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()

        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Volume" not in df.columns:
            df["Volume"] = np.nan

        return df.dropna(subset=["Open", "High", "Low", "Close"]), ""

    except Exception as exc:
        return pd.DataFrame(), f"Yahoo Finance request failed for {yf_symbol}: {exc}"


# ============================================================
# Indicators
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 8, slow: int = 21, signal: int = 5):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA_9"] = ema(out["Close"], 9)
    out["EMA_21"] = ema(out["Close"], 21)
    out["EMA_50"] = ema(out["Close"], 50)
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["SMA_200"] = sma(out["Close"], 200)
    out["RSI"] = rsi(out["Close"], 9)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["ATR"] = atr(out, 14)
    out["ADX"] = adx(out, 14)
    out["BB_UPPER"], out["BB_MID"], out["BB_LOWER"] = bollinger(out["Close"])

    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    volume = out["Volume"].fillna(0)
    if volume.replace(0, np.nan).dropna().empty:
        volume = pd.Series(1.0, index=out.index)
    day_key = pd.to_datetime(out.index).date
    out["VWAP"] = (typical * volume).groupby(day_key).cumsum() / volume.groupby(day_key).cumsum().replace(0, np.nan)

    return out


# ============================================================
# Analysis
# ============================================================

def analyse_symbol(symbol: str, period: str, interval: str) -> dict:
    df, err = load_yahoo_data(symbol, period, interval)
    if err:
        return {"symbol": symbol, "mapped_symbol": normalize_yahoo_symbol(symbol), "error": err}
    if len(df) < 80:
        return {"symbol": symbol, "mapped_symbol": normalize_yahoo_symbol(symbol), "error": f"Limited candle history: {len(df)} candles."}

    df = add_indicators(df).dropna()
    if df.empty:
        return {"symbol": symbol, "mapped_symbol": normalize_yahoo_symbol(symbol), "error": "Not enough clean indicator data."}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(latest["Close"])
    score = 0
    notes = []

    if latest["EMA_9"] > latest["EMA_21"]:
        score += 1
        notes.append("EMA 9 above EMA 21")
    else:
        score -= 1
        notes.append("EMA 9 below EMA 21")

    if pd.notna(latest["SMA_50"]) and price > latest["SMA_50"]:
        score += 1
        notes.append("Price above SMA50")
    elif pd.notna(latest["SMA_50"]):
        score -= 1
        notes.append("Price below SMA50")

    if price > latest["VWAP"]:
        score += 1
        notes.append("Price above VWAP")
    else:
        score -= 1
        notes.append("Price below VWAP")

    if latest["RSI"] > 55:
        score += 1
        notes.append("RSI bullish")
    elif latest["RSI"] < 45:
        score -= 1
        notes.append("RSI bearish")

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 1
        notes.append("MACD bullish")
    else:
        score -= 1
        notes.append("MACD bearish")

    if prev["EMA_9"] <= prev["EMA_21"] and latest["EMA_9"] > latest["EMA_21"]:
        score += 2
        notes.append("Fresh bullish EMA cross")
    elif prev["EMA_9"] >= prev["EMA_21"] and latest["EMA_9"] < latest["EMA_21"]:
        score -= 2
        notes.append("Fresh bearish EMA cross")

    if pd.notna(latest["ADX"]):
        if latest["ADX"] >= 30:
            score += 1 if score > 0 else -1
            notes.append("Strong trend quality")
        elif latest["ADX"] < 18:
            notes.append("Weak/choppy trend quality")

    if score >= 4:
        bucket = "WATCH LONG"
        css = "good"
    elif score <= -4:
        bucket = "WATCH SHORT"
        css = "bad"
    elif abs(score) >= 2:
        bucket = "NEAR TRIGGER"
        css = "warn"
    else:
        bucket = "NO TRADE"
        css = "neutral"

    atr_pct = float(latest["ATR"] / price * 100) if price and pd.notna(latest["ATR"]) else np.nan

    return {
        "symbol": symbol,
        "mapped_symbol": normalize_yahoo_symbol(symbol),
        "bucket": bucket,
        "css": css,
        "score": round(float(score), 2),
        "last_price": price,
        "rsi": float(latest["RSI"]),
        "adx": float(latest["ADX"]) if pd.notna(latest["ADX"]) else np.nan,
        "atr_pct": atr_pct,
        "vwap": float(latest["VWAP"]),
        "notes": "; ".join(notes[:7]),
        "df": df,
    }


def format_number(value, decimals=4):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "—"


def format_r(value):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.2f}R"
    except Exception:
        return "—"



# ============================================================
# Trade statistics / expectancy frontend
# ============================================================

def closed_trades_dataframe(state: dict) -> pd.DataFrame:
    """Return closed trades as a clean dataframe for analytics."""
    closed = state.get("closed_trades", []) or []
    df = pd.DataFrame(closed)
    if df.empty:
        return df

    for col in ["r_result", "entry", "exit_price", "risk_cash", "risk_percent_used", "score", "cost_adjusted_rr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["opened_at", "closed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "r_result" in df.columns:
        df["equity_r"] = df["r_result"].fillna(0).cumsum()
        df["trade_no"] = np.arange(1, len(df) + 1)
        df["win"] = df["r_result"] > 0

    # Normalise optional columns produced by newer scanner versions.
    if "session" not in df.columns:
        df["session"] = "Unknown"
    if "volatility" not in df.columns:
        df["volatility"] = "Unknown"
    if "regime" not in df.columns:
        df["regime"] = "Unknown"
    if "exit_reason" not in df.columns:
        df["exit_reason"] = "Unknown"
    if "symbol" not in df.columns:
        df["symbol"] = "Unknown"
    if "direction" not in df.columns:
        df["direction"] = "Unknown"
    return df


def performance_summary(df: pd.DataFrame) -> dict:
    if df.empty or "r_result" not in df.columns:
        return {"total_closed": 0}
    r = pd.to_numeric(df["r_result"], errors="coerce").dropna()
    if r.empty:
        return {"total_closed": 0}
    wins = r[r > 0]
    losses = r[r <= 0]
    equity = r.cumsum()
    running_high = equity.cummax()
    drawdown = equity - running_high
    return {
        "total_closed": int(len(r)),
        "wins": int((r > 0).sum()),
        "losses": int((r <= 0).sum()),
        "win_rate": float((r > 0).mean() * 100),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "total_r": float(r.sum()),
        "best_r": float(r.max()),
        "worst_r": float(r.min()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.inf,
        "max_drawdown_r": float(drawdown.min()) if not drawdown.empty else 0.0,
        "expectancy_r": float(r.mean()),
    }


def group_performance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or "r_result" not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[group_col] = tmp[group_col].fillna("Unknown").astype(str)
    grouped = tmp.groupby(group_col, dropna=False).agg(
        Trades=("r_result", "count"),
        Win_Rate=("win", "mean"),
        Avg_R=("r_result", "mean"),
        Total_R=("r_result", "sum"),
        Best_R=("r_result", "max"),
        Worst_R=("r_result", "min"),
    ).reset_index()
    grouped["Win_Rate"] = grouped["Win_Rate"] * 100
    return grouped.sort_values(["Total_R", "Avg_R"], ascending=False)


def equity_curve_figure(df: pd.DataFrame):
    fig = go.Figure()
    if df.empty or "equity_r" not in df.columns:
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=35, b=20))
        return fig
    x = df["closed_at"] if "closed_at" in df.columns and df["closed_at"].notna().any() else df["trade_no"]
    fig.add_trace(go.Scatter(x=x, y=df["equity_r"], mode="lines+markers", name="Cumulative R"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        height=380,
        title="Equity curve in R-multiples",
        yaxis_title="Cumulative R",
        xaxis_title="Trade close time" if "closed_at" in df.columns and df["closed_at"].notna().any() else "Trade #",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def bar_performance_figure(grouped: pd.DataFrame, label_col: str, title: str):
    fig = go.Figure()
    if grouped.empty or "Total_R" not in grouped.columns:
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
        return fig
    fig.add_trace(go.Bar(x=grouped[label_col], y=grouped["Total_R"], name="Total R"))
    fig.update_layout(height=340, title=title, yaxis_title="Total R", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def scanner_latest_dataframe(state: dict) -> pd.DataFrame:
    latest = state.get("latest_scan", []) or []
    return pd.DataFrame(latest)


def open_trades_dataframe(state: dict) -> pd.DataFrame:
    """Return open trades as dataframe with risk/reward context."""
    open_trades = state.get("open_trades", []) or []
    df = pd.DataFrame(open_trades)
    if df.empty:
        return df
    numeric_cols = ["entry", "stop", "target", "risk_cash", "risk_percent_used", "suggested_units", "risk_per_unit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["opened_at", "last_checked_candle"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if {"entry", "stop", "target"}.issubset(df.columns):
        df["planned_rr"] = np.where(
            df["direction"].astype(str).str.upper().eq("LONG"),
            (df["target"] - df["entry"]) / (df["entry"] - df["stop"]).replace(0, np.nan),
            (df["entry"] - df["target"]) / (df["stop"] - df["entry"]).replace(0, np.nan),
        )
    return df


def enrich_open_trades_with_live_prices(open_df: pd.DataFrame, period: str, interval: str) -> pd.DataFrame:
    """Add live Yahoo reference price and current R estimate to open trades."""
    if open_df.empty:
        return open_df
    out = open_df.copy()
    current_prices = []
    current_r = []
    distance_to_stop_pct = []
    distance_to_target_pct = []
    for _, row in out.iterrows():
        symbol = str(row.get("symbol", ""))
        data, err = load_yahoo_data(symbol, "5d", interval)
        if err or data.empty:
            price = np.nan
        else:
            price = float(data.iloc[-1]["Close"])
        entry = row.get("entry", np.nan)
        stop = row.get("stop", np.nan)
        target = row.get("target", np.nan)
        risk = abs(entry - stop) if pd.notna(entry) and pd.notna(stop) else np.nan
        direction = str(row.get("direction", "")).upper()
        if pd.notna(price) and pd.notna(risk) and risk > 0:
            r_now = (price - entry) / risk if direction == "LONG" else (entry - price) / risk
        else:
            r_now = np.nan
        if pd.notna(price) and price != 0 and pd.notna(stop):
            stop_pct = abs(price - stop) / price * 100
        else:
            stop_pct = np.nan
        if pd.notna(price) and price != 0 and pd.notna(target):
            target_pct = abs(target - price) / price * 100
        else:
            target_pct = np.nan
        current_prices.append(price)
        current_r.append(r_now)
        distance_to_stop_pct.append(stop_pct)
        distance_to_target_pct.append(target_pct)
    out["current_price_ref"] = current_prices
    out["current_r_est"] = current_r
    out["distance_to_stop_pct"] = distance_to_stop_pct
    out["distance_to_target_pct"] = distance_to_target_pct
    return out


def insight_badge(label: str, value: str, detail: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="trade-card">
            <div class="trade-muted">{label}</div>
            <div class="big-status {tone}">{value}</div>
            <div class="trade-muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def best_worst_from_group(grouped: pd.DataFrame, label_col: str) -> tuple[str, str]:
    if grouped.empty:
        return "Not enough data", "Not enough data"
    best = grouped.sort_values(["Total_R", "Avg_R"], ascending=False).iloc[0]
    worst = grouped.sort_values(["Total_R", "Avg_R"], ascending=True).iloc[0]
    best_txt = f"{best[label_col]}: {best['Total_R']:.2f}R over {int(best['Trades'])} trade(s)"
    worst_txt = f"{worst[label_col]}: {worst['Total_R']:.2f}R over {int(worst['Trades'])} trade(s)"
    return best_txt, worst_txt


def build_performance_insights(closed_df: pd.DataFrame, open_df: pd.DataFrame) -> list[str]:
    insights = []
    if closed_df.empty or "r_result" not in closed_df.columns:
        insights.append("No closed-trade sample yet. The dashboard will become more useful after 20+ closed trades.")
        if not open_df.empty:
            insights.append(f"You currently have {len(open_df)} open trade(s); focus on execution and exits before judging performance.")
        return insights

    summary = performance_summary(closed_df)
    n = summary.get("total_closed", 0)
    expectancy = summary.get("expectancy_r", 0)
    drawdown = summary.get("max_drawdown_r", 0)
    win_rate = summary.get("win_rate", 0)

    if n < 20:
        insights.append(f"Sample size is still small: {n} closed trade(s). Use the numbers as early feedback, not final proof.")
    elif n < 50:
        insights.append(f"Sample size is improving: {n} closed trade(s). Patterns are useful, but still avoid overfitting.")
    else:
        insights.append(f"Sample size is meaningful: {n} closed trade(s). Asset/session filters now deserve attention.")

    if expectancy > 0.15:
        insights.append(f"Expectancy is strong at {expectancy:.2f}R per trade. The system is currently paying you per setup.")
    elif expectancy > 0:
        insights.append(f"Expectancy is positive but thin at {expectancy:.2f}R. Costs, discipline, and avoiding weak sessions matter a lot.")
    else:
        insights.append(f"Expectancy is negative at {expectancy:.2f}R. Reduce trade frequency and focus on the best-performing conditions.")

    if win_rate >= 55:
        insights.append(f"Win rate is healthy at {win_rate:.1f}%. Check whether winners are still large enough versus losers.")
    elif win_rate >= 40:
        insights.append(f"Win rate is moderate at {win_rate:.1f}%. This can work if average winners are meaningfully bigger than losers.")
    else:
        insights.append(f"Win rate is low at {win_rate:.1f}%. Only acceptable if your winners are much larger than losses.")

    if drawdown <= -5:
        insights.append(f"Max drawdown is {drawdown:.2f}R. Consider lowering risk temporarily until the equity curve stabilizes.")
    elif drawdown <= -2:
        insights.append(f"Drawdown is controlled but visible at {drawdown:.2f}R. Watch whether it clusters around one asset/session.")
    else:
        insights.append(f"Drawdown is currently mild at {drawdown:.2f}R.")

    for col, name in [("symbol", "asset"), ("session", "session"), ("volatility", "volatility regime"), ("regime", "setup regime")]:
        g = group_performance(closed_df, col)
        if not g.empty and len(g) >= 2:
            best, worst = best_worst_from_group(g, col)
            insights.append(f"Best {name}: {best}. Weakest {name}: {worst}.")

    if not open_df.empty:
        risk_total = pd.to_numeric(open_df.get("risk_cash", pd.Series(dtype=float)), errors="coerce").sum()
        insights.append(f"Open exposure: {len(open_df)} open trade(s), about {risk_total:.2f} account-currency risk based on your JSON settings.")
    return insights[:9]


def drawdown_figure(df: pd.DataFrame):
    fig = go.Figure()
    if df.empty or "r_result" not in df.columns:
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
        return fig
    r = pd.to_numeric(df["r_result"], errors="coerce").fillna(0)
    equity = r.cumsum()
    dd = equity - equity.cummax()
    x = df["closed_at"] if "closed_at" in df.columns and df["closed_at"].notna().any() else df.get("trade_no", np.arange(1, len(df) + 1))
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", name="Drawdown R", fill="tozeroy"))
    fig.update_layout(height=330, title="Drawdown curve", yaxis_title="Drawdown in R", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def win_loss_distribution_figure(df: pd.DataFrame):
    fig = go.Figure()
    if df.empty or "r_result" not in df.columns:
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
        return fig
    fig.add_trace(go.Histogram(x=pd.to_numeric(df["r_result"], errors="coerce"), nbinsx=20, name="R distribution"))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=330, title="Winner/loser distribution", xaxis_title="Trade result in R", yaxis_title="Count", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def rolling_expectancy_figure(df: pd.DataFrame, window: int = 10):
    fig = go.Figure()
    if df.empty or "r_result" not in df.columns:
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
        return fig
    tmp = df.copy()
    tmp["rolling_expectancy"] = pd.to_numeric(tmp["r_result"], errors="coerce").rolling(window, min_periods=3).mean()
    x = tmp["closed_at"] if "closed_at" in tmp.columns and tmp["closed_at"].notna().any() else tmp.get("trade_no", np.arange(1, len(tmp) + 1))
    fig.add_trace(go.Scatter(x=x, y=tmp["rolling_expectancy"], mode="lines+markers", name=f"Rolling {window}-trade expectancy"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=330, title=f"Rolling {window}-trade expectancy", yaxis_title="R/trade", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def latest_scan_summary(scanner_df: pd.DataFrame) -> str:
    if scanner_df.empty:
        return "No current scanner rows yet."
    if "Bucket" in scanner_df.columns:
        actionable = scanner_df[scanner_df["Bucket"].astype(str).str.contains("WATCH|ACT", case=False, na=False)]
        near = scanner_df[scanner_df["Bucket"].astype(str).str.contains("NEAR", case=False, na=False)]
        return f"{len(actionable)} watch/action candidate(s), {len(near)} near-trigger candidate(s), {len(scanner_df)} symbol(s) checked."
    return f"{len(scanner_df)} latest scanner row(s) available."


def plot_chart(df: pd.DataFrame, symbol: str, chart_settings: dict):
    show_rsi = chart_settings.get("show_rsi", True)
    show_macd = chart_settings.get("show_macd", True)
    show_volume = chart_settings.get("show_volume", False)

    rows = 1
    row_titles = [f"{symbol} price"]
    row_heights = [0.62]

    rsi_row = None
    macd_row = None
    volume_row = None

    if show_rsi:
        rows += 1
        rsi_row = rows
        row_titles.append("RSI")
        row_heights.append(0.16)

    if show_macd:
        rows += 1
        macd_row = rows
        row_titles.append("MACD")
        row_heights.append(0.18)

    if show_volume:
        rows += 1
        volume_row = rows
        row_titles.append("Volume")
        row_heights.append(0.14)

    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=tuple(row_titles),
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles",
    ), row=1, col=1)

    overlays = [
        ("show_ema_9", "EMA_9", "EMA 9"),
        ("show_ema_21", "EMA_21", "EMA 21"),
        ("show_ema_50", "EMA_50", "EMA 50"),
        ("show_sma_20", "SMA_20", "SMA 20"),
        ("show_sma_50", "SMA_50", "SMA 50"),
        ("show_sma_200", "SMA_200", "SMA 200"),
        ("show_vwap", "VWAP", "VWAP"),
        ("show_bb_upper", "BB_UPPER", "BB Upper"),
        ("show_bb_mid", "BB_MID", "BB Mid"),
        ("show_bb_lower", "BB_LOWER", "BB Lower"),
    ]

    for toggle, col, name in overlays:
        if chart_settings.get(toggle, False) and col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name), row=1, col=1)

    if rsi_row:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=rsi_row, col=1)
        if chart_settings.get("show_rsi_levels", True):
            fig.add_hline(y=70, line_dash="dash", row=rsi_row, col=1)
            fig.add_hline(y=30, line_dash="dash", row=rsi_row, col=1)

    if macd_row:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=macd_row, col=1)
        if chart_settings.get("show_macd_hist", True):
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram"), row=macd_row, col=1)

    if volume_row and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=volume_row, col=1)

    fig.update_layout(
        height=820 if rows > 1 else 680,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_header(selected: dict | None, state: dict):
    if selected and "error" not in selected:
        css = selected.get("css", "neutral")
        headline = selected.get("bucket", "WAIT")
        sub = selected.get("notes", "No notes.")
        symbol = selected.get("symbol", "—")
        price = format_number(selected.get("last_price"))
    else:
        css = "neutral"
        headline = "Dashboard ready"
        sub = "Choose a symbol and run analysis."
        symbol = "—"
        price = "—"

    cols = st.columns([1.7, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(
            f"""
            <div class="trade-card">
                <div class="trade-muted">{symbol} · Yahoo Finance data</div>
                <div class="big-status {css}">{headline}</div>
                <div class="trade-muted">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cols[1].metric("Last price", price)
    cols[2].metric("Open trades", len(state.get("open_trades", [])))
    cols[3].metric("Closed trades", len(state.get("closed_trades", [])))
    cols[4].metric("State updated", state.get("last_updated", "—")[:19] if state.get("last_updated") else "—")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Trading style")
    mode = st.selectbox("Mode", ["Day Trader", "Swing Trader", "Custom"], index=0)

    st.header("Yahoo Finance analysis")
    default_watchlist = "EURUSD, XAUUSD, US100, US500, AAPL"
    watchlist_text = st.text_area("Watchlist symbols", default_watchlist)
    selected_symbol = st.text_input("Detailed symbol", "XAUUSD")

    period = st.selectbox("Lookback period", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Candle interval", ["1m", "5m", "15m", "30m", "60m", "1d"], index=1)

    st.header("Chart indicators")
    st.caption("Turn chart overlays and lower panels on/off.")

    show_ema_9 = st.checkbox("EMA 9", value=True)
    show_ema_21 = st.checkbox("EMA 21", value=True)
    show_ema_50 = st.checkbox("EMA 50", value=False)
    show_sma_20 = st.checkbox("SMA 20", value=False)
    show_sma_50 = st.checkbox("SMA 50", value=False)
    show_sma_200 = st.checkbox("SMA 200", value=False)
    show_vwap = st.checkbox("VWAP", value=True)

    show_bollinger = st.checkbox("Bollinger Bands", value=False)
    show_rsi = st.checkbox("RSI panel", value=True)
    show_rsi_levels = st.checkbox("RSI 30/70 levels", value=True)
    show_macd = st.checkbox("MACD panel", value=True)
    show_macd_hist = st.checkbox("MACD histogram", value=True)
    show_volume = st.checkbox("Volume panel", value=False)

    chart_settings = {
        "show_ema_9": show_ema_9,
        "show_ema_21": show_ema_21,
        "show_ema_50": show_ema_50,
        "show_sma_20": show_sma_20,
        "show_sma_50": show_sma_50,
        "show_sma_200": show_sma_200,
        "show_vwap": show_vwap,
        "show_bb_upper": show_bollinger,
        "show_bb_mid": show_bollinger,
        "show_bb_lower": show_bollinger,
        "show_rsi": show_rsi,
        "show_rsi_levels": show_rsi_levels,
        "show_macd": show_macd,
        "show_macd_hist": show_macd_hist,
        "show_volume": show_volume,
    }

    st.header("Display")
    show_raw_state = st.checkbox("Show raw GitHub state", value=False)

    st.divider()
    st.warning("Educational decision-support dashboard only. It does not place broker trades or send Telegram messages.")


# ============================================================
# Main app
# ============================================================

st.title("📈 Trade Decision Cockpit")
st.caption("Cockpit-style dashboard using Yahoo Finance for live/on-demand analysis and GitHub for trade-state persistence.")

if st.button("🔄 Refresh all data"):
    st.cache_data.clear()
    st.rerun()

state = load_trade_state_from_github()
if "error" in state:
    st.error(state["error"])
    st.stop()

detail_result = analyse_symbol(selected_symbol, period, interval)
render_header(None if "error" in detail_result else detail_result, state)

scanner_tab, detail_tab, active_tab, stats_tab, diagnostics_tab = st.tabs([
    "1 · Market Scanner",
    "2 · Trade Detail",
    "3 · Active Trades",
    "4 · Statistics",
    "5 · Settings & Diagnostics",
])


with scanner_tab:
    st.subheader("Market scanner")
    st.caption("On-demand scanner. Uses Yahoo Finance. It analyses only when the dashboard is opened/refreshed.")

    symbols = [s.strip() for s in watchlist_text.replace("\n", ",").split(",") if s.strip()]
    rows = []

    for sym in symbols:
        result = analyse_symbol(sym, period, interval)
        if "error" in result:
            rows.append({
                "Bucket": "ERROR",
                "Symbol": sym,
                "Yahoo symbol": normalize_yahoo_symbol(sym),
                "Score": np.nan,
                "Last price": np.nan,
                "RSI": np.nan,
                "ADX": np.nan,
                "ATR %": np.nan,
                "Notes": result["error"],
            })
        else:
            rows.append({
                "Bucket": result["bucket"],
                "Symbol": sym,
                "Yahoo symbol": result["mapped_symbol"],
                "Score": result["score"],
                "Last price": result["last_price"],
                "RSI": result["rsi"],
                "ADX": result["adx"],
                "ATR %": result["atr_pct"],
                "Notes": result["notes"],
            })

    scanner_df = pd.DataFrame(rows)
    if not scanner_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Watch long", int((scanner_df["Bucket"] == "WATCH LONG").sum()))
        c2.metric("Watch short", int((scanner_df["Bucket"] == "WATCH SHORT").sum()))
        c3.metric("Near trigger", int((scanner_df["Bucket"] == "NEAR TRIGGER").sum()))
        c4.metric("No trade", int((scanner_df["Bucket"] == "NO TRADE").sum()))

        order = {"WATCH LONG": 1, "WATCH SHORT": 1, "NEAR TRIGGER": 2, "NO TRADE": 3, "ERROR": 4}
        scanner_df["_order"] = scanner_df["Bucket"].map(order).fillna(9)
        scanner_df = scanner_df.sort_values(["_order", "Score"], ascending=[True, False]).drop(columns=["_order"])
        st.dataframe(scanner_df, use_container_width=True, hide_index=True)

    st.caption("Your Render scanner/trade engine still handles ACT NOW and trade tracking. This Streamlit dashboard is read-only/on-demand.")


with detail_tab:
    st.subheader("Trade detail")

    if "error" in detail_result:
        st.error(detail_result["error"])
    else:
        df = detail_result["df"]

        a, b, c, d = st.columns(4)
        a.metric("Signal bucket", detail_result["bucket"])
        b.metric("Score", detail_result["score"])
        c.metric("RSI", format_number(detail_result["rsi"], 2))
        d.metric("ATR %", format_number(detail_result["atr_pct"], 3))

        st.markdown("### Technical summary")
        st.write(detail_result["notes"])

        st.plotly_chart(
            plot_chart(df, f"{detail_result['symbol']} ({detail_result['mapped_symbol']})", chart_settings),
            use_container_width=True,
        )

        latest = df.iloc[-1]
        snap = pd.DataFrame({
            "Indicator": ["Close", "EMA 9", "EMA 21", "EMA 50", "SMA 20", "SMA 50", "SMA 200", "VWAP", "RSI", "MACD", "MACD Signal", "ATR", "ADX"],
            "Value": [
                latest.get("Close"),
                latest.get("EMA_9"),
                latest.get("EMA_21"),
                latest.get("EMA_50"),
                latest.get("SMA_20"),
                latest.get("SMA_50"),
                latest.get("SMA_200"),
                latest.get("VWAP"),
                latest.get("RSI"),
                latest.get("MACD"),
                latest.get("MACD_SIGNAL"),
                latest.get("ATR"),
                latest.get("ADX"),
            ],
        })
        st.dataframe(snap, use_container_width=True, hide_index=True)


with active_tab:
    st.subheader("Active trades")
    st.caption("Live read-only frontend of what the Render scanner is currently tracking from your GitHub trade-state JSON.")

    open_df_raw = open_trades_dataframe(state)
    closed_df = closed_trades_dataframe(state)

    if open_df_raw.empty:
        st.info("No open trades currently recorded.")
    else:
        open_df = enrich_open_trades_with_live_prices(open_df_raw, period, interval)
        total_risk = pd.to_numeric(open_df.get("risk_cash", pd.Series(dtype=float)), errors="coerce").sum()
        avg_live_r = pd.to_numeric(open_df.get("current_r_est", pd.Series(dtype=float)), errors="coerce").mean()
        closest_stop = pd.to_numeric(open_df.get("distance_to_stop_pct", pd.Series(dtype=float)), errors="coerce").min()
        best_open = pd.to_numeric(open_df.get("current_r_est", pd.Series(dtype=float)), errors="coerce").max()

        a, b, c, d = st.columns(4)
        a.metric("Open trades", len(open_df))
        b.metric("Open risk", format_number(total_risk, 2))
        c.metric("Avg open R", format_r(avg_live_r))
        d.metric("Closest stop distance", f"{closest_stop:.2f}%" if pd.notna(closest_stop) else "—")

        if pd.notna(best_open) and best_open >= 1:
            st.success("At least one open trade is above +1R. Watch the trailing-stop/breakeven logic closely.")
        elif pd.notna(avg_live_r) and avg_live_r < -0.5:
            st.warning("Open trades are currently under pressure. Check whether several positions are exposed to the same market move.")
        else:
            st.info("Open-trade risk is being tracked. Current R values are estimates from Yahoo reference prices, not broker fills.")

        for trade in open_df.to_dict("records"):
            r_now = trade.get("current_r_est", np.nan)
            tone = "good" if pd.notna(r_now) and r_now > 0 else "bad" if pd.notna(r_now) and r_now < -0.5 else "neutral"
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
                c1.markdown(f"### {trade.get('symbol', '—')}")
                c1.caption(f"{trade.get('direction', '—')} · {trade.get('status', 'OPEN')}")
                c2.metric("Entry", format_number(trade.get("entry")))
                c2.caption(f"Now: {format_number(trade.get('current_price_ref'))}")
                c3.metric("Current R", format_r(r_now))
                c3.caption(f"Planned RR: {format_r(trade.get('planned_rr'))}")
                c4.metric("Stop", format_number(trade.get("stop")))
                c4.caption(f"Distance: {format_number(trade.get('distance_to_stop_pct'), 2)}%")
                c5.metric("Target", format_number(trade.get("target")))
                c5.caption(f"Distance: {format_number(trade.get('distance_to_target_pct'), 2)}%")
                st.markdown(f"<span class='{tone}'>Risk cash: {format_number(trade.get('risk_cash'), 2)} · Risk used: {format_number(trade.get('risk_percent_used'), 2)}%</span>", unsafe_allow_html=True)
                if trade.get("notes"):
                    st.write(trade.get("notes"))

        preferred_open = [
            "symbol", "direction", "entry", "current_price_ref", "current_r_est", "stop", "target",
            "planned_rr", "risk_cash", "risk_percent_used", "opened_at", "last_checked_candle",
        ]
        cols = [c for c in preferred_open if c in open_df.columns]
        with st.expander("Open trade table"):
            st.dataframe(open_df[cols] if cols else open_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Closed trades overview")
    if closed_df.empty:
        st.info("No closed trades recorded yet.")
    else:
        summary = performance_summary(closed_df)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total R", format_r(summary.get("total_r")))
        m2.metric("Expectancy", format_r(summary.get("expectancy_r")))
        m3.metric("Win rate", f"{summary.get('win_rate', 0):.1f}%")
        pf = summary.get("profit_factor")
        m4.metric("Profit factor", "∞" if pf == np.inf else format_number(pf, 2))
        m5.metric("Max DD", format_r(summary.get("max_drawdown_r")))

        preferred = [
            "symbol", "direction", "entry", "exit_price", "exit_reason", "r_result",
            "session", "volatility", "regime", "opened_at", "closed_at",
        ]
        cols = [c for c in preferred if c in closed_df.columns]
        st.dataframe(closed_df[cols] if cols else closed_df, use_container_width=True, hide_index=True)


with stats_tab:
    st.subheader("Insight cockpit")
    st.caption("This is the decision layer: what your trade book says about edge, risk, weak spots, and the conditions worth prioritising.")

    closed_df = closed_trades_dataframe(state)
    open_df_raw = open_trades_dataframe(state)
    latest_df = scanner_latest_dataframe(state)

    if closed_df.empty or "r_result" not in closed_df.columns:
        st.info("No closed trades with R results yet. Once your scanner closes trades, this page will automatically become your performance cockpit.")
        if not open_df_raw.empty:
            st.markdown("### Current open-trade context")
            st.dataframe(open_df_raw, use_container_width=True, hide_index=True)
    else:
        summary = performance_summary(closed_df)
        insights = build_performance_insights(closed_df, open_df_raw)

        st.markdown("### Executive readout")
        c1, c2, c3 = st.columns(3)
        expectancy = summary.get("expectancy_r", 0)
        dd = summary.get("max_drawdown_r", 0)
        pf = summary.get("profit_factor")
        with c1:
            insight_badge(
                "System edge",
                format_r(expectancy),
                "Average R per closed trade. Positive means the rules are currently paying you.",
                "good" if expectancy > 0.1 else "warn" if expectancy > 0 else "bad",
            )
        with c2:
            insight_badge(
                "Risk pressure",
                format_r(dd),
                "Worst cumulative pullback in R. This tells you how hard the system has hurt during bad sequences.",
                "bad" if dd <= -5 else "warn" if dd <= -2 else "good",
            )
        with c3:
            insight_badge(
                "Trade quality",
                "∞" if pf == np.inf else format_number(pf, 2),
                "Profit factor. Above 1 means winners outweigh losers in total R.",
                "good" if pf == np.inf or pf > 1.3 else "warn" if pf > 1 else "bad",
            )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Closed", summary.get("total_closed", 0))
        k2.metric("Win rate", f"{summary.get('win_rate', 0):.1f}%")
        k3.metric("Total R", format_r(summary.get("total_r")))
        k4.metric("Avg R", format_r(summary.get("avg_r")))
        k5.metric("Best", format_r(summary.get("best_r")))
        k6.metric("Worst", format_r(summary.get("worst_r")))

        st.markdown("### Key insights")
        for item in insights:
            st.write(f"• {item}")

        st.markdown("### Equity, drawdown, and consistency")
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(equity_curve_figure(closed_df), use_container_width=True)
        with p2:
            st.plotly_chart(drawdown_figure(closed_df), use_container_width=True)
        p3, p4 = st.columns(2)
        with p3:
            st.plotly_chart(rolling_expectancy_figure(closed_df, window=10), use_container_width=True)
        with p4:
            st.plotly_chart(win_loss_distribution_figure(closed_df), use_container_width=True)

        st.markdown("### Where the edge is coming from")
        by_symbol = group_performance(closed_df, "symbol")
        by_session = group_performance(closed_df, "session")
        by_vol = group_performance(closed_df, "volatility")
        by_regime = group_performance(closed_df, "regime")
        by_exit = group_performance(closed_df, "exit_reason")
        by_direction = group_performance(closed_df, "direction")

        t1, t2 = st.columns(2)
        with t1:
            st.plotly_chart(bar_performance_figure(by_symbol, "symbol", "Total R by asset"), use_container_width=True)
            st.dataframe(by_symbol, use_container_width=True, hide_index=True)
        with t2:
            st.plotly_chart(bar_performance_figure(by_session, "session", "Total R by session"), use_container_width=True)
            st.dataframe(by_session, use_container_width=True, hide_index=True)

        t3, t4 = st.columns(2)
        with t3:
            st.plotly_chart(bar_performance_figure(by_vol, "volatility", "Total R by volatility regime"), use_container_width=True)
            st.dataframe(by_vol, use_container_width=True, hide_index=True)
        with t4:
            st.plotly_chart(bar_performance_figure(by_regime, "regime", "Total R by setup regime"), use_container_width=True)
            st.dataframe(by_regime, use_container_width=True, hide_index=True)

        st.markdown("### Execution and exit diagnosis")
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(bar_performance_figure(by_exit, "exit_reason", "Total R by exit reason"), use_container_width=True)
            st.dataframe(by_exit, use_container_width=True, hide_index=True)
        with e2:
            st.plotly_chart(bar_performance_figure(by_direction, "direction", "Total R by direction"), use_container_width=True)
            st.dataframe(by_direction, use_container_width=True, hide_index=True)

        st.markdown("### Actionable rules to consider")
        rules = []
        for grouped, col_name, label in [
            (by_symbol, "symbol", "asset"),
            (by_session, "session", "session"),
            (by_vol, "volatility", "volatility regime"),
            (by_regime, "regime", "setup regime"),
        ]:
            if not grouped.empty:
                weak = grouped[(grouped["Trades"] >= 3) & (grouped["Total_R"] < 0)].sort_values("Total_R").head(3)
                strong = grouped[(grouped["Trades"] >= 3) & (grouped["Total_R"] > 0)].sort_values("Total_R", ascending=False).head(3)
                for _, row in strong.iterrows():
                    rules.append(f"Prioritise {label} **{row[col_name]}**: {row['Total_R']:.2f}R over {int(row['Trades'])} trades.")
                for _, row in weak.iterrows():
                    rules.append(f"Be stricter with {label} **{row[col_name]}**: {row['Total_R']:.2f}R over {int(row['Trades'])} trades.")
        if rules:
            for r in rules[:10]:
                st.write(f"• {r}")
        else:
            st.info("Not enough repeated conditions yet for rule recommendations. Aim for at least 3 trades per category.")

        st.markdown("### Recent trade journal")
        preferred = [
            "trade_no", "symbol", "direction", "r_result", "equity_r", "exit_reason",
            "session", "volatility", "regime", "opened_at", "closed_at",
        ]
        cols = [c for c in preferred if c in closed_df.columns]
        st.dataframe(closed_df[cols].tail(40), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Latest scanner state")
    if latest_df.empty:
        st.info("No latest_scan section found yet. Your scanner can write latest_scan into the GitHub state JSON so this dashboard shows the exact latest Render scan.")
    else:
        st.write(latest_scan_summary(latest_df))
        st.dataframe(latest_df, use_container_width=True, hide_index=True)


with diagnostics_tab:
    st.subheader("Settings & diagnostics")
    st.write("This dashboard reads trade state from GitHub and live/on-demand candles from Yahoo Finance.")
    st.write(f"Selected mode: **{mode}**")
    st.write(f"Period / interval: **{period} / {interval}**")
    st.write(f"GitHub state file: **{github_config().get('repo')}/{github_config().get('file')}**")
    st.write("Gold mapping: **XAUUSD → GC=F**")

    latest_scan = state.get("latest_scan", [])
    st.subheader("Latest scan from Render scanner")
    if latest_scan:
        st.dataframe(pd.DataFrame(latest_scan), use_container_width=True)
    else:
        st.info("No latest_scan section found yet. Your Render scanner can be upgraded to write latest_scan into the GitHub state file.")

    if show_raw_state:
        st.subheader("Raw GitHub state")
        st.json(state)
