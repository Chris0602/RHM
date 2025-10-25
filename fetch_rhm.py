import json, time, os
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "RHM.DE"
LOOKBACK_DAYS = 180

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, window=20, mult=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + mult*std
    lower = ma - mult*std
    return ma, upper, lower

def chaikin_mf(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    mf_vol = mf_mult * df['Volume']
    cmf = mf_vol.rolling(period).sum() / df['Volume'].rolling(period).sum()
    return cmf

def obv(df):
    obv_vals = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_vals.append(obv_vals[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_vals.append(obv_vals[-1] - df['Volume'].iloc[i])
        else:
            obv_vals.append(obv_vals[-1])
    return pd.Series(obv_vals, index=df.index)

def main():
    df = yf.download(TICKER, period=f"{LOOKBACK_DAYS}d", interval="1d", auto_adjust=False)
    if df.empty:
        raise SystemExit("Keine Daten geladen.")

    df = df.rename(columns=str.title)  # Open/High/Low/Close/Adj Close/Volume
    # Indikatoren
    df['BB_MA20'], df['BB_Upper'], df['BB_Lower'] = bollinger(df['Close'], 20, 2)
    df['RSI14'] = rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
    df['CMF20'] = chaikin_mf(df, 20)
    df['OBV'] = obv(df)

    # Unterstützungen/Widerstände (einfach: jüngste Swing-Punkte)
    recent = df.tail(40).copy()
    support = float(recent['Low'].rolling(5).min().tail(1))
    resistance = float(recent['High'].rolling(5).max().tail(1))

    # Historische Volatilität (20-Tage) als ATM-IV-Proxy, wenn echte IV fehlt
    # (Annualisiert; Hinweis: Das ist HV, nicht echte IV!)
    logret = np.log(df['Close']).diff()
    hv20 = float(logret.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(logret.dropna())>20 else None

    latest = df.tail(1).iloc[0]
    payload = {
        "ticker": TICKER,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "price": {
            "close": float(latest['Close']),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "volume": int(latest['Volume'])
        },
        "indicators": {
            "bollinger": {
                "ma20": float(latest['BB_MA20']) if not np.isnan(latest['BB_MA20']) else None,
                "upper": float(latest['BB_Upper']) if not np.isnan(latest['BB_Upper']) else None,
                "lower": float(latest['BB_Lower']) if not np.isnan(latest['BB_Lower']) else None
            },
            "rsi14": float(latest['RSI14']) if not np.isnan(latest['RSI14']) else None,
            "macd": {
                "line": float(latest['MACD']) if not np.isnan(latest['MACD']) else None,
                "signal": float(latest['MACD_Signal']) if not np.isnan(latest['MACD_Signal']) else None,
                "hist": float(latest['MACD_Hist']) if not np.isnan(latest['MACD_Hist']) else None
            },
            "cmf20": float(latest['CMF20']) if not np.isnan(latest['CMF20']) else None,
            "obv": float(latest['OBV']) if not np.isnan(latest['OBV']) else None
        },
        "levels": {
            "support_hint": round(support, 2),
            "resistance_hint": round(resistance, 2)
        },
        "volatility": {
            "hv20_proxy": round(hv20, 4) if hv20 else None,
            "iv_atm_estimate_note": "Wenn echte ATM-IV (Eurex) verfügbar, hier einfügen; aktuell HV20 als Proxy."
        }
    }

    os.makedirs("data", exist_ok=True)
    with open("data/rhm_latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    df.tail(120).to_csv("data/rhm_prices_last120.csv")

if __name__ == "__main__":
    main()
