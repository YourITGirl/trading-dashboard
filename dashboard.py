import base64
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(page_title='Trade Decision Cockpit', page_icon='📈', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
.trade-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 18px; padding: 1rem 1.1rem; background: rgba(128,128,128,0.06); margin-bottom: .75rem;}
.big-status {font-size: 1.45rem; font-weight: 800; line-height: 1.2; margin-bottom: .35rem;}
.trade-muted {opacity: .75; font-size: .92rem;}
.good {color: #16a34a;} .warn {color: #d97706;} .bad {color: #dc2626;} .neutral {color: #64748b;}
</style>
''', unsafe_allow_html=True)


def get_secret(name, default=''):
    try: return str(st.secrets.get(name, default))
    except Exception: return default


def github_config():
    return {
        'token': get_secret('GITHUB_STATE_TOKEN'),
        'owner': get_secret('GITHUB_STATE_OWNER'),
        'repo': get_secret('GITHUB_STATE_REPO'),
        'branch': get_secret('GITHUB_STATE_BRANCH', 'main'),
        'file': get_secret('GITHUB_STATE_FILE', 'trade_manager_open_trades.json'),
    }


def twelve_key(): return get_secret('TWELVE_DATA_API_KEY')


@st.cache_data(ttl=60)
def load_trade_state_from_github():
    cfg = github_config()
    missing = [k for k, v in cfg.items() if not v and k != 'branch']
    if missing:
        return {'error': f"Missing Streamlit secret(s): {', '.join(missing)}", 'open_trades': [], 'closed_trades': [], 'latest_scan': []}
    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['file']}?ref={cfg['branch']}"
    headers = {'Authorization': f"Bearer {cfg['token']}", 'Accept': 'application/vnd.github+json'}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if not resp.ok:
            return {'error': f'GitHub API error {resp.status_code}: {resp.text[:300]}', 'open_trades': [], 'closed_trades': [], 'latest_scan': []}
        decoded = base64.b64decode(resp.json().get('content', '')).decode('utf-8')
        data = json.loads(decoded)
        data.setdefault('open_trades', []); data.setdefault('closed_trades', []); data.setdefault('latest_scan', []); data.setdefault('last_updated', '')
        return data
    except Exception as exc:
        return {'error': f'Could not load GitHub state file: {exc}', 'open_trades': [], 'closed_trades': [], 'latest_scan': []}

SYMBOL_MAP = {'EURUSD':'EUR/USD','GBPUSD':'GBP/USD','USDJPY':'USD/JPY','USDCHF':'USD/CHF','AUDUSD':'AUD/USD','USDCAD':'USD/CAD','NZDUSD':'NZD/USD','XAUUSD':'XAU/USD','GOLD':'XAU/USD','US100':'NAS100','NAS100':'NAS100','US500':'SPX500','SPX500':'SPX500'}

def normalize_twelve_symbol(symbol): return SYMBOL_MAP.get(str(symbol).strip().upper().replace(' ', ''), str(symbol).strip().upper())

@st.cache_data(ttl=120)
def load_twelvedata(symbol, interval, outputsize=500):
    api_key = twelve_key()
    if not api_key: return pd.DataFrame(), 'Missing Streamlit secret: TWELVE_DATA_API_KEY'
    params = {'symbol': normalize_twelve_symbol(symbol), 'interval': interval, 'outputsize': int(outputsize), 'apikey': api_key, 'format': 'JSON', 'order': 'ASC'}
    try:
        resp = requests.get('https://api.twelvedata.com/time_series', params=params, timeout=20)
        if not resp.ok: return pd.DataFrame(), f'TwelveData HTTP error {resp.status_code}: {resp.text[:220]}'
        payload = resp.json()
        if payload.get('status') == 'error': return pd.DataFrame(), f"TwelveData error: {payload.get('message', 'unknown error')}"
        values = payload.get('values', [])
        if not values: return pd.DataFrame(), 'No candles returned.'
        df = pd.DataFrame(values)
        df['Datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df = df.dropna(subset=['Datetime']).set_index('Datetime')
        df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Volume' not in df.columns: df['Volume'] = np.nan
        return df[['Open','High','Low','Close','Volume']].dropna(subset=['Open','High','Low','Close']), ''
    except Exception as exc:
        return pd.DataFrame(), f'TwelveData request failed: {exc}'


def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=9):
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=8, slow=21, signal=5):
    m = ema(series, fast) - ema(series, slow); s = ema(m, signal); return m, s, m-s

def atr(df, period=14):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    dx = ((plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def add_indicators(df):
    out = df.copy()
    out['EMA_FAST'] = ema(out['Close'], 9); out['EMA_SLOW'] = ema(out['Close'], 21)
    out['SMA_50'] = out['Close'].rolling(50).mean(); out['SMA_200'] = out['Close'].rolling(200).mean()
    out['RSI'] = rsi(out['Close'], 9); out['MACD'], out['MACD_SIGNAL'], out['MACD_HIST'] = macd(out['Close'])
    out['ATR'] = atr(out, 14); out['ADX'] = adx(out, 14)
    typical = (out['High'] + out['Low'] + out['Close']) / 3
    volume = out['Volume'].fillna(0)
    if volume.replace(0, np.nan).dropna().empty: volume = pd.Series(1.0, index=out.index)
    day_key = pd.to_datetime(out.index).date
    out['VWAP'] = (typical * volume).groupby(day_key).cumsum() / volume.groupby(day_key).cumsum().replace(0, np.nan)
    return out

def analyse_symbol(symbol, interval, outputsize):
    df, err = load_twelvedata(symbol, interval, outputsize)
    if err: return {'symbol': symbol, 'error': err}
    if len(df) < 80: return {'symbol': symbol, 'error': f'Limited candle history: {len(df)} candles.'}
    df = add_indicators(df).dropna()
    if df.empty: return {'symbol': symbol, 'error': 'Not enough clean indicator data.'}
    latest = df.iloc[-1]; prev = df.iloc[-2]; price = float(latest['Close'])
    score = 0; notes = []
    checks = [
        ('EMA trend bullish', latest['EMA_FAST'] > latest['EMA_SLOW'], 1, 'EMA trend bearish'),
        ('Price above SMA50', pd.notna(latest['SMA_50']) and price > latest['SMA_50'], 1, 'Price below SMA50'),
        ('Price above VWAP', price > latest['VWAP'], 1, 'Price below VWAP'),
        ('RSI bullish', latest['RSI'] > 55, 1, 'RSI bearish' if latest['RSI'] < 45 else 'RSI neutral'),
        ('MACD bullish', latest['MACD'] > latest['MACD_SIGNAL'], 1, 'MACD bearish'),
    ]
    for good, cond, pts, bad in checks:
        if cond: score += pts; notes.append(good)
        else:
            if 'neutral' not in bad.lower(): score -= pts
            notes.append(bad)
    if prev['EMA_FAST'] <= prev['EMA_SLOW'] and latest['EMA_FAST'] > latest['EMA_SLOW']:
        score += 2; notes.append('Fresh bullish EMA cross')
    elif prev['EMA_FAST'] >= prev['EMA_SLOW'] and latest['EMA_FAST'] < latest['EMA_SLOW']:
        score -= 2; notes.append('Fresh bearish EMA cross')
    if pd.notna(latest['ADX']) and latest['ADX'] >= 30:
        score += 1 if score > 0 else -1; notes.append('Strong trend quality')
    bucket, css = ('WATCH LONG','good') if score >= 4 else ('WATCH SHORT','bad') if score <= -4 else ('NEAR TRIGGER','warn') if abs(score) >= 2 else ('NO TRADE','neutral')
    atr_pct = float(latest['ATR'] / price * 100) if price and pd.notna(latest['ATR']) else np.nan
    return {'symbol': symbol, 'mapped_symbol': normalize_twelve_symbol(symbol), 'bucket': bucket, 'css': css, 'score': round(float(score),2), 'last_price': price, 'rsi': float(latest['RSI']), 'adx': float(latest['ADX']) if pd.notna(latest['ADX']) else np.nan, 'atr_pct': atr_pct, 'vwap': float(latest['VWAP']), 'notes': '; '.join(notes[:7]), 'df': df}

def fmt(value, decimals=4):
    try:
        if value is None or pd.isna(value): return '—'
        return f'{float(value):,.{decimals}f}'
    except Exception: return '—'

def fmt_r(value):
    try:
        if value is None or pd.isna(value): return '—'
        return f'{float(value):.2f}R'
    except Exception: return '—'

def plot_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.04, row_heights=[.62,.18,.20], subplot_titles=(f'{symbol} price','RSI','MACD'))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'), row=1, col=1)
    for col, name in [('EMA_FAST','EMA 9'),('EMA_SLOW','EMA 21'),('VWAP','VWAP')]: fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1); fig.add_hline(y=70, line_dash='dash', row=2, col=1); fig.add_hline(y=30, line_dash='dash', row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'), row=3, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], mode='lines', name='Signal'), row=3, col=1); fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='Histogram'), row=3, col=1)
    fig.update_layout(height=780, xaxis_rangeslider_visible=False, legend_orientation='h', margin=dict(l=20,r=20,t=50,b=20))
    return fig

def render_header(selected, state):
    if selected and 'error' not in selected:
        css = selected.get('css','neutral'); headline = selected.get('bucket','WAIT'); sub = selected.get('notes','No notes.'); symbol = selected.get('symbol','—'); price = fmt(selected.get('last_price'))
    else:
        css='neutral'; headline='Dashboard ready'; sub='Choose a symbol and run analysis.'; symbol='—'; price='—'
    cols = st.columns([1.7,1,1,1,1])
    with cols[0]:
        st.markdown(f'''<div class="trade-card"><div class="trade-muted">{symbol} · on-demand analysis</div><div class="big-status {css}">{headline}</div><div class="trade-muted">{sub}</div></div>''', unsafe_allow_html=True)
    cols[1].metric('Last price', price); cols[2].metric('Open trades', len(state.get('open_trades', []))); cols[3].metric('Closed trades', len(state.get('closed_trades', []))); cols[4].metric('State updated', state.get('last_updated','—')[:19] if state.get('last_updated') else '—')

with st.sidebar:
    st.header('Trading style')
    mode = st.selectbox('Mode', ['Day Trader', 'Swing Trader', 'Custom'], index=0)
    st.caption('Day Trader defaults: 5-minute candles and fast technical alignment.')
    st.header('Live analysis')
    watchlist_text = st.text_area('Watchlist symbols', 'EURUSD, XAUUSD, US100, US500, AAPL, NVDA, TSLA')
    selected_symbol = st.text_input('Detailed symbol', 'XAUUSD')
    interval = st.selectbox('Candle interval', ['1min','5min','15min','30min','1h','1day'], index=1)
    candles = st.selectbox('Candles', [100,300,500,1000], index=1)
    show_raw_state = st.checkbox('Show raw GitHub state', value=False)
    st.divider(); st.warning('Educational decision-support dashboard only. It does not place broker trades or send Telegram messages.')

st.title('📈 Trade Decision Cockpit')
st.caption('Cockpit-style dashboard: market scanner, trade detail, active trades, and diagnostics. No Telegram logic.')
if st.button('🔄 Refresh all data'):
    st.cache_data.clear(); st.rerun()
state = load_trade_state_from_github()
if 'error' in state: st.error(state['error']); st.stop()
if not twelve_key(): st.warning('Missing Streamlit secret: TWELVE_DATA_API_KEY. Live scanner/detail analysis will not work until you add it.')
detail_result = analyse_symbol(selected_symbol, interval, candles) if twelve_key() else {'symbol': selected_symbol, 'error': 'Missing TwelveData key.'}
render_header(None if 'error' in detail_result else detail_result, state)
scanner_tab, detail_tab, active_tab, diagnostics_tab = st.tabs(['1 · Market Scanner','2 · Trade Detail','3 · Active Trades','4 · Settings & Diagnostics'])

with scanner_tab:
    st.subheader('Market scanner')
    st.caption('On-demand scanner. It analyses only when the dashboard is opened/refreshed, not constantly.')
    symbols = [s.strip() for s in watchlist_text.replace('\n', ',').split(',') if s.strip()]
    rows = []
    for sym in symbols:
        result = analyse_symbol(sym, interval, candles) if twelve_key() else {'symbol': sym, 'error': 'Missing TwelveData key.'}
        if 'error' in result:
            rows.append({'Bucket':'ERROR','Symbol':sym,'Mapped':normalize_twelve_symbol(sym),'Score':np.nan,'Last price':np.nan,'RSI':np.nan,'ADX':np.nan,'ATR %':np.nan,'Notes':result['error']})
        else:
            rows.append({'Bucket':result['bucket'],'Symbol':sym,'Mapped':result['mapped_symbol'],'Score':result['score'],'Last price':result['last_price'],'RSI':result['rsi'],'ADX':result['adx'],'ATR %':result['atr_pct'],'Notes':result['notes']})
    scanner_df = pd.DataFrame(rows)
    if not scanner_df.empty:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Watch long', int((scanner_df['Bucket']=='WATCH LONG').sum()))
        c2.metric('Watch short', int((scanner_df['Bucket']=='WATCH SHORT').sum()))
        c3.metric('Near trigger', int((scanner_df['Bucket']=='NEAR TRIGGER').sum()))
        c4.metric('No trade', int((scanner_df['Bucket']=='NO TRADE').sum()))
        order = {'WATCH LONG':1,'WATCH SHORT':1,'NEAR TRIGGER':2,'NO TRADE':3,'ERROR':4}
        scanner_df['_order'] = scanner_df['Bucket'].map(order).fillna(9)
        scanner_df = scanner_df.sort_values(['_order','Score'], ascending=[True,False]).drop(columns=['_order'])
        st.dataframe(scanner_df, use_container_width=True, hide_index=True)
    st.caption('ACT NOW remains handled by your Render scanner/trade engine. This dashboard is visual/on-demand analysis.')

with detail_tab:
    st.subheader('Trade detail')
    if 'error' in detail_result: st.error(detail_result['error'])
    else:
        df = detail_result['df']; a,b,c,d = st.columns(4)
        a.metric('Signal bucket', detail_result['bucket']); b.metric('Score', detail_result['score']); c.metric('RSI', fmt(detail_result['rsi'],2)); d.metric('ATR %', fmt(detail_result['atr_pct'],3))
        st.markdown('### Technical summary'); st.write(detail_result['notes'])
        st.plotly_chart(plot_chart(df, detail_result['symbol']), use_container_width=True)
        latest = df.iloc[-1]
        snap = pd.DataFrame({'Indicator':['Close','EMA 9','EMA 21','VWAP','RSI','MACD','MACD Signal','ATR','ADX'], 'Value':[latest.get('Close'),latest.get('EMA_FAST'),latest.get('EMA_SLOW'),latest.get('VWAP'),latest.get('RSI'),latest.get('MACD'),latest.get('MACD_SIGNAL'),latest.get('ATR'),latest.get('ADX')]})
        st.dataframe(snap, use_container_width=True, hide_index=True)

with active_tab:
    st.subheader('Active trades')
    st.caption('Read-only view from your private GitHub trade-state file.')
    open_trades = state.get('open_trades', []); closed_trades = state.get('closed_trades', [])
    if not open_trades: st.info('No open trades currently recorded.')
    else:
        for trade in open_trades:
            with st.container(border=True):
                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(f"### {trade.get('symbol','—')}"); c1.caption(f"{trade.get('direction','—')} · {trade.get('status','OPEN')}")
                c2.metric('Entry', fmt(trade.get('entry'))); c2.caption(f"Opened: {trade.get('opened_at','—')}")
                c3.metric('Stop', fmt(trade.get('stop'))); c3.metric('Target', fmt(trade.get('target')))
                c4.metric('Risk cash', fmt(trade.get('risk_cash'),2)); c4.caption(f"Risk used: {fmt(trade.get('risk_percent_used'),2)}%")
                if trade.get('notes'): st.write(trade.get('notes'))
    st.divider(); st.subheader('Closed trades')
    closed_df = pd.DataFrame(closed_trades)
    if closed_df.empty: st.info('No closed trades recorded yet.')
    else:
        if 'r_result' in closed_df.columns:
            closed_df['r_result_num'] = pd.to_numeric(closed_df['r_result'], errors='coerce')
            total_r = closed_df['r_result_num'].sum(); avg_r = closed_df['r_result_num'].mean(); win_rate = (closed_df['r_result_num'].gt(0).sum() / closed_df['r_result_num'].notna().sum() * 100) if closed_df['r_result_num'].notna().sum() else 0
            m1,m2,m3 = st.columns(3); m1.metric('Total R', fmt_r(total_r)); m2.metric('Average R', fmt_r(avg_r)); m3.metric('Win rate', f'{win_rate:.1f}%')
        preferred = ['symbol','direction','entry','exit_price','exit_reason','r_result','opened_at','closed_at']
        cols = [c for c in preferred if c in closed_df.columns]
        st.dataframe(closed_df[cols] if cols else closed_df, use_container_width=True, hide_index=True)

with diagnostics_tab:
    st.subheader('Settings & diagnostics')
    st.write('This dashboard reads trade state from GitHub and live candles from TwelveData.')
    st.write(f'Selected mode: **{mode}**'); st.write(f'Interval: **{interval}**'); st.write(f'Candles: **{candles}**')
    st.write(f"GitHub state file: **{github_config().get('repo')}/{github_config().get('file')}**")
    st.subheader('Latest scan from Render scanner')
    latest_scan = state.get('latest_scan', [])
    if latest_scan: st.dataframe(pd.DataFrame(latest_scan), use_container_width=True)
    else: st.info('No latest_scan section found yet. Your Render scanner can later be upgraded to write latest_scan into the GitHub state file.')
    if show_raw_state:
        st.subheader('Raw GitHub state'); st.json(state)
