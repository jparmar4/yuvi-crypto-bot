import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Bot
from telegram.constants import ParseMode
import asyncio
import time
from datetime import datetime
import csv

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TOKEN)

# ==== CONFIG ====
COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
INTERVALS = ['1h', '15m']
COOLDOWNS = {'1h': 3600, '15m': 900}  # seconds
LOG_FILE = "signal_log.csv"

def fetch_data(symbol, interval, limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    try:
        response = requests.get(url)
        data = response.json()
        print(f"[{symbol}-{interval}] Fetched data rows:", len(data))
        if not data or len(data) < 35:
            return None
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower

def macd_crossover(macd_line, signal_line):
    if len(macd_line) < 2 or len(signal_line) < 2:
        return None
    prev = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr = macd_line.iloc[-1] - signal_line.iloc[-1]
    if prev < 0 and curr > 0:
        return "bullish"
    elif prev > 0 and curr < 0:
        return "bearish"
    else:
        return None

def generate_signal(df):
    if df is None or len(df) < 35:
        return None
    ma_short = df['close'].rolling(window=7).mean().iloc[-1]
    ma_long = df['close'].rolling(window=25).mean().iloc[-1]
    rsi = calculate_rsi(df['close']).iloc[-1]
    macd_line, signal_line, hist = calculate_macd(df['close'])
    macd_now, macd_signal_now, macd_hist_now = macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
    bb_upper_now, bb_mid_now, bb_lower_now = bb_upper.iloc[-1], bb_mid.iloc[-1], bb_lower.iloc[-1]
    trend_signal = "LONG 🚀" if ma_short > ma_long else "SHORT 📉"
    rsi_state = "Overbought (Possible Short)" if rsi > 70 else ("Oversold (Possible Long)" if rsi < 30 else None)
    macd_cross = macd_crossover(macd_line, signal_line)
    return {
        'trend_signal': trend_signal,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'rsi': rsi,
        'rsi_state': rsi_state,
        'macd_now': macd_now,
        'macd_signal_now': macd_signal_now,
        'macd_hist_now': macd_hist_now,
        'macd_cross': macd_cross,
        'bb_upper_now': bb_upper_now,
        'bb_mid_now': bb_mid_now,
        'bb_lower_now': bb_lower_now,
        'macd_line': macd_line,
        'signal_line': signal_line,
        'df': df,
    }

def get_last_signal(symbol, interval):
    fname = f"last_signal_{symbol}_{interval}.txt"
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return f.read().strip()
    return None

def set_last_signal(symbol, interval, signal):
    fname = f"last_signal_{symbol}_{interval}.txt"
    with open(fname, 'w') as f:
        f.write(signal)

def get_last_alert_time(symbol, interval):
    fname = f"last_alert_time_{symbol}_{interval}.txt"
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return float(f.read())
    return 0

def set_last_alert_time(symbol, interval, timestamp):
    fname = f"last_alert_time_{symbol}_{interval}.txt"
    with open(fname, 'w') as f:
        f.write(str(timestamp))

def save_chart(df, signal, symbol, interval, rsi, macd_line, signal_line, bb_upper, bb_lower):
    plt.figure(figsize=(14, 8))
    plt.plot(df['open_time'], df['close'], label='Close Price', color='blue')
    plt.plot(df['open_time'], bb_upper, label='BB Upper', color='green', linestyle='--')
    plt.plot(df['open_time'], bb_lower, label='BB Lower', color='red', linestyle='--')
    plt.title(f"{symbol} ({interval}) - Signal: {signal} - RSI: {rsi:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.grid(True)
    plt.legend(loc='upper left')
    ax2 = plt.gca().twinx()
    ax2.plot(df['open_time'], macd_line, label='MACD', color='magenta', alpha=0.5)
    ax2.plot(df['open_time'], signal_line, label='MACD Signal', color='orange', alpha=0.5)
    ax2.legend(loc='lower right')
    filename = f"{symbol}_{interval}_chart.png"
    plt.savefig(filename)
    plt.close()
    return filename

def log_signal(symbol, interval, signal, rsi, macd_cross, price):
    log_data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "rsi": round(rsi, 2) if rsi is not None else "",
        "macd_cross": macd_cross if macd_cross else "",
        "price": round(price, 2) if price is not None else ""
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

async def send_signal_to_telegram(message, chart_path=None):
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.HTML)
    if chart_path:
        with open(chart_path, 'rb') as chart:
            await bot.send_photo(chat_id=CHAT_ID, photo=chart)

async def main():
    now = time.time()
    for symbol in COINS:
        for interval in INTERVALS:
            cooldown = COOLDOWNS.get(interval, 900)
            last_alert_time = get_last_alert_time(symbol, interval)
            if now - last_alert_time < cooldown:
                print(f"{symbol} ({interval}): Cooldown active, skipping signal.")
                continue
            data = generate_signal(fetch_data(symbol, interval))
            if data is None:
                print(f"{symbol} ({interval}): Not enough data.")
                continue
            # Only alert if RSI is overbought/oversold or MACD crossover
            if data['rsi_state'] is None and data['macd_cross'] is None:
                print(f"{symbol} ({interval}): No RSI or MACD crossover alert.")
                continue
            last_signal = get_last_signal(symbol, interval)
            significant_event = False
            alert_msg = ""
            if last_signal != data['trend_signal']:
                significant_event = True
                set_last_signal(symbol, interval, data['trend_signal'])
                alert_msg += f"Trend changed! New trend: <b>{data['trend_signal']}</b>\n"
            if data['macd_cross']:
                significant_event = True
                alert_msg += f"MACD Crossover: <b>{data['macd_cross'].capitalize()} Crossover</b>\n"
            if not significant_event:
                print(f"{symbol} ({interval}): No new significant signal.")
                continue
            chart_path = save_chart(
                data['df'], data['trend_signal'], symbol, interval, data['rsi'],
                data['macd_line'], data['signal_line'],
                data['df']['close'].rolling(window=20).mean() + 2*data['df']['close'].rolling(window=20).std(),
                data['df']['close'].rolling(window=20).mean() - 2*data['df']['close'].rolling(window=20).std()
            )
            msg = (
                f"<b>{symbol} ({interval}) Alert</b>\n"
                f"{alert_msg}"
                f"RSI (14): <b>{data['rsi']:.2f}</b> ({data['rsi_state'] if data['rsi_state'] else 'Neutral'})\n"
                f"Short MA (7): <b>{data['ma_short']:.2f}</b>\n"
                f"Long MA (25): <b>{data['ma_long']:.2f}</b>\n"
                f"MACD: <b>{data['macd_now']:.2f}</b>, MACD Signal: <b>{data['macd_signal_now']:.2f}</b>\n"
                f"Bollinger Bands: <b>{data['bb_upper_now']:.2f} / {data['bb_mid_now']:.2f} / {data['bb_lower_now']:.2f}</b>\n"
            )
            await send_signal_to_telegram(msg, chart_path=chart_path)
            set_last_alert_time(symbol, interval, now)
            log_signal(
                symbol, interval, data['trend_signal'], data['rsi'], data['macd_cross'], data['df']['close'].iloc[-1]
            )
            print(f"{symbol} ({interval}): Alert sent and logged.")

if __name__ == "__main__":
    asyncio.run(main())
