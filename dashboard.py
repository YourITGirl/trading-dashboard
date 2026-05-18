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
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA_FAST"] = ema(out["Close"], 9)
    out["EMA_SLOW"] = ema(out["Close"], 21)
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    out["RSI"] = rsi(out["Close"], 9)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["ATR"] = atr(out, 14)
    out["ADX"] = adx(out, 14)

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

    if latest["EMA_FAST"] > latest["EMA_SLOW"]:
        score += 1
        notes.append("Fast EMA above slow EMA")
    else:
        score -= 1
        notes.append("Fast EMA below slow EMA")

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

    if prev["EMA_FAST"] <= prev["EMA_SLOW"] and latest["EMA_FAST"] > latest["EMA_SLOW"]:
        score += 2
        notes.append("Fresh bullish EMA cross")
    elif prev["EMA_FAST"] >= prev["EMA_SLOW"] and latest["EMA_FAST"] < latest["EMA_SLOW"]:
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


def plot_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=(f"{symbol} price", "RSI", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles",
    ), row=1, col=1)

    for col, name in [("EMA_FAST", "EMA 9"), ("EMA_SLOW", "EMA 21"), ("VWAP", "VWAP")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram"), row=3, col=1)

    fig.update_layout(
        height=780,
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
    default_watchlist = "EURUSD, XAUUSD, US100, US500, AAPL, NVDA, TSLA"
    watchlist_text = st.text_area("Watchlist symbols", default_watchlist)
    selected_symbol = st.text_input("Detailed symbol", "XAUUSD")

    period = st.selectbox("Lookback period", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Candle interval", ["1m", "5m", "15m", "30m", "60m", "1d"], index=1)

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

scanner_tab, detail_tab, active_tab, diagnostics_tab = st.tabs([
    "1 · Market Scanner",
    "2 · Trade Detail",
    "3 · Active Trades",
    "4 · Settings & Diagnostics",
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

        st.plotly_chart(plot_chart(df, f"{detail_result['symbol']} ({detail_result['mapped_symbol']})"), use_container_width=True)

        latest = df.iloc[-1]
        snap = pd.DataFrame({
            "Indicator": ["Close", "EMA 9", "EMA 21", "VWAP", "RSI", "MACD", "MACD Signal", "ATR", "ADX"],
            "Value": [
                latest.get("Close"),
                latest.get("EMA_FAST"),
                latest.get("EMA_SLOW"),
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
    st.caption("Read-only view from your private GitHub trade-state file.")

    open_trades = state.get("open_trades", [])
    closed_trades = state.get("closed_trades", [])

    if not open_trades:
        st.info("No open trades currently recorded.")
    else:
        for trade in open_trades:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"### {trade.get('symbol', '—')}")
                c1.caption(f"{trade.get('direction', '—')} · {trade.get('status', 'OPEN')}")
                c2.metric("Entry", format_number(trade.get("entry")))
                c2.caption(f"Opened: {trade.get('opened_at', '—')}")
                c3.metric("Stop", format_number(trade.get("stop")))
                c3.metric("Target", format_number(trade.get("target")))
                c4.metric("Risk cash", format_number(trade.get("risk_cash"), 2))
                c4.caption(f"Risk used: {format_number(trade.get('risk_percent_used'), 2)}%")
                if trade.get("notes"):
                    st.write(trade.get("notes"))

    st.divider()
    st.subheader("Closed trades")
    closed_df = pd.DataFrame(closed_trades)
    if closed_df.empty:
        st.info("No closed trades recorded yet.")
    else:
        if "r_result" in closed_df.columns:
            closed_df["r_result_num"] = pd.to_numeric(closed_df["r_result"], errors="coerce")
            total_r = closed_df["r_result_num"].sum()
            avg_r = closed_df["r_result_num"].mean()
            win_rate = (closed_df["r_result_num"].gt(0).sum() / closed_df["r_result_num"].notna().sum() * 100) if closed_df["r_result_num"].notna().sum() else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total R", format_r(total_r))
            m2.metric("Average R", format_r(avg_r))
            m3.metric("Win rate", f"{win_rate:.1f}%")

        preferred = ["symbol", "direction", "entry", "exit_price", "exit_reason", "r_result", "opened_at", "closed_at"]
        cols = [c for c in preferred if c in closed_df.columns]
        st.dataframe(closed_df[cols] if cols else closed_df, use_container_width=True, hide_index=True)


with diagnostics_tab:
    st.subheader("Settings & diagnostics")
    st.write("This dashboard reads trade state from GitHub and live/on-demand candles from Yahoo Finance.")
    st.write(f"Selected mode: **{mode}**")
    st.write(f"Period / interval: **{period} / {interval}**")
    st.write(f"GitHub state file: **{github_config().get('repo')}/{github_config().get('file')}**")

    latest_scan = state.get("latest_scan", [])
    st.subheader("Latest scan from Render scanner")
    if latest_scan:
        st.dataframe(pd.DataFrame(latest_scan), use_container_width=True)
    else:
        st.info("No latest_scan section found yet. Your Render scanner can be upgraded to write latest_scan into the GitHub state file.")

    if show_raw_state:
        st.subheader("Raw GitHub state")
        st.json(state)
