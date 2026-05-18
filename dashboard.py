import base64
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trading Assistant Dashboard", page_icon="📊", layout="wide")

YAHOO_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "XAUUSD": "XAUUSD=X",
    "GOLD": "XAUUSD=X",
    "GC": "GC=F",
    "US100": "NQ=F",
    "NAS100": "NQ=F",
    "NASDAQ100": "NQ=F",
    "US500": "ES=F",
    "SPX500": "ES=F",
    "SP500": "ES=F",
    "S&P500": "ES=F",
    "VIX": "^VIX",
}


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
        return {"error": f"Missing Streamlit secret(s): {', '.join(missing)}", "open_trades": [], "closed_trades": []}

    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['file']}?ref={cfg['branch']}"
    headers = {"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if not response.ok:
            return {"error": f"GitHub API error {response.status_code}: {response.text[:300]}", "open_trades": [], "closed_trades": []}

        payload = response.json()
        decoded = base64.b64decode(payload.get("content", "")).decode("utf-8")
        data = json.loads(decoded)
        data.setdefault("open_trades", [])
        data.setdefault("closed_trades", [])
        data.setdefault("latest_scan", [])
        data.setdefault("last_updated", "")
        return data
    except Exception as exc:
        return {"error": f"Could not load GitHub state file: {exc}", "open_trades": [], "closed_trades": []}


def normalize_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace(" ", "")
    return YAHOO_SYMBOL_MAP.get(s, symbol.strip().upper())


def yahoo_period_for_interval(interval: str) -> str:
    if interval == "1m":
        return "7d"
    if interval in {"5m", "15m", "30m", "60m"}:
        return "30d"
    return "1y"


def normalize_yahoo_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    needed = ["open", "high", "low", "close"]
    if not all(c in out.columns for c in needed):
        return pd.DataFrame()
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=needed)


@st.cache_data(ttl=120)
def load_yahoo_data(symbol: str, interval: str = "5m", period: str | None = None) -> tuple[pd.DataFrame, str]:
    yahoo_symbol = normalize_yahoo_symbol(symbol)
    period = period or yahoo_period_for_interval(interval)
    try:
        df = yf.download(yahoo_symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        df = normalize_yahoo_df(df)
        if df.empty:
            return pd.DataFrame(), f"No Yahoo Finance candles returned for {symbol} mapped to {yahoo_symbol}."
        return df, ""
    except Exception as exc:
        return pd.DataFrame(), f"Yahoo Finance request failed for {symbol}: {exc}"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


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
    return macd_line, signal_line, macd_line - signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_9"] = ema(out["close"], 9)
    out["ema_21"] = ema(out["close"], 21)
    out["sma_50"] = out["close"].rolling(50).mean()
    out["rsi_9"] = rsi(out["close"], 9)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["atr_14"] = atr(out)
    typical = (out["high"] + out["low"] + out["close"]) / 3
    volume = out["volume"].fillna(0)
    if volume.replace(0, np.nan).dropna().empty:
        volume = pd.Series(1.0, index=out.index)
    day_key = pd.to_datetime(out.index).date
    out["vwap"] = (typical * volume).groupby(day_key).cumsum() / volume.groupby(day_key).cumsum().replace(0, np.nan)
    return out


def analyze_latest(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {"bias": "Not enough data", "score": 0, "reason": "Need more candles."}

    ind = add_indicators(df).dropna()
    if ind.empty or len(ind) < 2:
        return {"bias": "Not enough clean data", "score": 0, "reason": "Indicators could not be calculated."}

    last, prev = ind.iloc[-1], ind.iloc[-2]
    score, reasons = 0, []

    checks = [
        (last["ema_9"] > last["ema_21"], "EMA trend bullish", "EMA trend bearish"),
        (last["close"] > last["sma_50"], "Price above SMA50", "Price below SMA50"),
        (last["close"] > last["vwap"], "Price above VWAP", "Price below VWAP"),
        (last["macd"] > last["macd_signal"], "MACD bullish", "MACD bearish"),
    ]
    for ok, good, bad in checks:
        score += 1 if ok else -1
        reasons.append(good if ok else bad)

    if last["rsi_9"] > 55:
        score += 1
        reasons.append("RSI bullish")
    elif last["rsi_9"] < 45:
        score -= 1
        reasons.append("RSI bearish")

    if prev["ema_9"] <= prev["ema_21"] and last["ema_9"] > last["ema_21"]:
        score += 2
        reasons.append("Fresh bullish EMA cross")
    elif prev["ema_9"] >= prev["ema_21"] and last["ema_9"] < last["ema_21"]:
        score -= 2
        reasons.append("Fresh bearish EMA cross")

    bias = "Bullish" if score >= 4 else "Bearish" if score <= -4 else "Neutral / mixed"
    atr_pct = (last["atr_14"] / last["close"] * 100) if last["close"] else np.nan

    return {
        "bias": bias,
        "score": round(float(score), 2),
        "last_price": float(last["close"]),
        "rsi": float(last["rsi_9"]),
        "atr_pct": float(atr_pct) if pd.notna(atr_pct) else np.nan,
        "vwap": float(last["vwap"]),
        "reason": "; ".join(reasons[:7]),
    }


def clear_cache_and_reload():
    st.cache_data.clear()


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


def to_dataframe(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


st.title("📊 Trading Assistant Dashboard")
st.caption("Trade-state monitor + on-demand Yahoo Finance analysis. No Telegram messages are sent from this dashboard.")

top_left, top_right = st.columns([1, 1])
with top_left:
    if st.button("🔄 Refresh dashboard"):
        clear_cache_and_reload()
        st.rerun()

state = load_trade_state_from_github()
if "error" in state:
    st.error(state["error"])
    st.stop()

open_trades = state.get("open_trades", [])
closed_trades = state.get("closed_trades", [])
latest_scan = state.get("latest_scan", [])
last_updated = state.get("last_updated", "")

with top_right:
    st.caption(f"Last state update: {last_updated or 'Not available'}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Open trades", str(len(open_trades)))
col2.metric("Closed trades", str(len(closed_trades)))
col3.metric("Active symbols", str(len({t.get('symbol', '') for t in open_trades if t.get('symbol')})))
col4.metric("Live data", "Yahoo Finance")

tabs = st.tabs(["Open Trades", "Closed Trades", "Latest Scan", "Yahoo Analysis", "Raw State"])

with tabs[0]:
    st.subheader("Open Trades")
    if not open_trades:
        st.info("No open trades currently recorded.")
    else:
        for trade in open_trades:
            with st.container(border=True):
                h1, h2, h3, h4 = st.columns(4)
                h1.markdown(f"### {trade.get('symbol', 'Unknown')}")
                h1.caption(f"{trade.get('direction', 'Unknown')} · {trade.get('status', 'OPEN')}")
                h2.metric("Entry", format_number(trade.get("entry")))
                h2.caption(f"Opened: {trade.get('opened_at', '—')}")
                h3.metric("Stop", format_number(trade.get("stop")))
                h3.metric("Target", format_number(trade.get("target")))
                h4.metric("Risk cash", format_number(trade.get("risk_cash"), 2))
                h4.caption(f"Risk used: {format_number(trade.get('risk_percent_used'), 2)}%")
                if trade.get("notes"):
                    st.write(trade.get("notes"))

with tabs[1]:
    st.subheader("Closed Trades")
    closed_df = to_dataframe(closed_trades)
    if closed_df.empty:
        st.info("No closed trades recorded yet.")
    else:
        if "r_result" in closed_df.columns:
            closed_df["r_result_num"] = pd.to_numeric(closed_df["r_result"], errors="coerce")
            total_r = closed_df["r_result_num"].sum()
            avg_r = closed_df["r_result_num"].mean()
            wins = (closed_df["r_result_num"] > 0).sum()
            total = closed_df["r_result_num"].notna().sum()
            win_rate = (wins / total * 100) if total else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Total R", format_r(total_r))
            c2.metric("Average R", format_r(avg_r))
            c3.metric("Win rate", f"{win_rate:.1f}%")
        preferred_cols = ["symbol", "direction", "entry", "exit_price", "exit_reason", "r_result", "opened_at", "closed_at"]
        display_cols = [c for c in preferred_cols if c in closed_df.columns]
        st.dataframe(closed_df[display_cols] if display_cols else closed_df, use_container_width=True)

with tabs[2]:
    st.subheader("Latest Scan")
    if not latest_scan:
        st.warning("No latest_scan section found yet. To show all assets here, the scanner should write latest_scan into the GitHub state file.")
    else:
        st.dataframe(to_dataframe(latest_scan), use_container_width=True)

with tabs[3]:
    st.subheader("Yahoo Finance Analysis")
    st.caption("No TwelveData or Alpaca keys are required. Gold defaults to XAUUSD=X; futures can be checked with GC=F.")

    watchlist_text = st.text_input("Symbols to analyse", value="EURUSD, XAUUSD, US100, US500")
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m", "1d"], index=1)
    period = c2.selectbox("Period", ["auto", "1d", "5d", "7d", "30d", "60d", "6mo", "1y"], index=0)
    run_live = c3.button("Analyse Yahoo data")

    if run_live:
        symbols_to_check = [s.strip() for s in watchlist_text.replace("\n", ",").split(",") if s.strip()]
        rows = []
        selected_period = None if period == "auto" else period

        with st.spinner("Loading Yahoo Finance data..."):
            for sym in symbols_to_check:
                df, err = load_yahoo_data(sym, interval=interval, period=selected_period)
                if err:
                    rows.append({"symbol": sym, "mapped_symbol": normalize_yahoo_symbol(sym), "bias": "Error", "score": np.nan, "last_price": np.nan, "rsi": np.nan, "atr_pct": np.nan, "vwap": np.nan, "reason": err})
                    continue
                rows.append({"symbol": sym, "mapped_symbol": normalize_yahoo_symbol(sym), **analyze_latest(df)})

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)

        successful_symbols = [r["symbol"] for r in rows if r.get("bias") != "Error"]
        if successful_symbols:
            selected_symbol = st.selectbox("Chart symbol", successful_symbols)
            chart_df, chart_err = load_yahoo_data(selected_symbol, interval=interval, period=selected_period)
            if chart_err:
                st.error(chart_err)
            elif not chart_df.empty:
                ind = add_indicators(chart_df)
                chart_cols = [c for c in ["close", "ema_9", "ema_21", "vwap"] if c in ind.columns]
                st.line_chart(ind[chart_cols])

with tabs[4]:
    st.subheader("Raw GitHub State")
    st.json(state)
