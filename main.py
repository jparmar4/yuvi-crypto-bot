from dotenv import load_dotenv
load_dotenv()
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Bot
from telegram.constants import ParseMode
import asyncio
import time
from datetime import datetime, timezone
import csv

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TOKEN)

# ==== CONFIG ====
COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
INTERVALS = ['1h']
COOLDOWNS = {'1h': 3600}
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
        df['volume'] = df['volume'].astype(float)
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

def ma_crossover(ma_short, ma_long, prev_ma_short, prev_ma_long):
    if prev_ma_short < prev_ma_long and ma_short > ma_long:
        return "bullish"
    elif prev_ma_short > prev_ma_long and ma_short < ma_long:
        return "bearish"
    return None

def is_volume_spike(df, n=20, threshold=2):
    recent_volumes = df['volume'][-n:]
    avg_volume = recent_volumes.mean()
    current_volume = df['volume'].iloc[-1]
    return current_volume > threshold * avg_volume

def calculate_trade_levels(entry_price, trend, support, resistance):
    if trend == "LONG 🚀":
        tp1 = entry_price * 1.005
        tp2 = entry_price * 1.01
        tp3 = entry_price * 1.015
        tp4 = entry_price * 1.02
        stop_loss = support * 0.99
        sl_text = f"SL below support: {stop_loss:.2f}"
        instructions = "Enter after bullish confirmation near entry, SL below support. Adjust size and always manage risk."
    else:
        tp1 = entry_price * 0.995
        tp2 = entry_price * 0.99
        tp3 = entry_price * 0.985
        tp4 = entry_price * 0.98
        stop_loss = resistance * 1.01
        sl_text = f"SL above resistance: {stop_loss:.2f}"
        instructions = "Enter after bearish confirmation near entry, SL above resistance. Adjust size and always manage risk."
    return tp1, tp2, tp3, tp4, stop_loss, sl_text, instructions

def calculate_fibonacci_levels(swing_low, swing_high, trend):
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    if trend == "LONG 🚀":
        for r in fib_ratios:
            levels[f"Fib {int(r*100)}%"] = swing_low + (swing_high - swing_low) * r
    else:
        for r in fib_ratios:
            levels[f"Fib {int(r*100)}%"] = swing_high - (swing_high - swing_low) * r
    return levels

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
        with open(fname, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def set_last_signal(symbol, interval, signal):
    fname = f"last_signal_{symbol}_{interval}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
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
    clean_signal = signal.replace("🚀", "").replace("📉", "").strip()
    plt.figure(figsize=(14, 8))
    plt.plot(df['open_time'], df['close'], label='Close Price', color='blue')
    plt.plot(df['open_time'], bb_upper, label='BB Upper', color='green', linestyle='--')
    plt.plot(df['open_time'], bb_lower, label='BB Lower', color='red', linestyle='--')
    plt.title(f"{symbol} ({interval}) - Signal: {clean_signal} - RSI: {rsi:.2f}")
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
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "rsi": round(rsi, 2) if rsi is not None else "",
        "macd_cross": macd_cross if macd_cross else "",
        "price": round(price, 2) if price is not None else ""
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline='', encoding='utf-8') as f:
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

            prev_ma_short = data['df']['close'].rolling(window=7).mean().iloc[-2]
            prev_ma_long = data['df']['close'].rolling(window=25).mean().iloc[-2]
            ma_cross = ma_crossover(data['ma_short'], data['ma_long'], prev_ma_short, prev_ma_long)

            entry_price = float(data['df']['close'].iloc[-1])
            recent_prices = data['df']['close'][-20:]
            support = float(recent_prices.min())
            resistance = float(recent_prices.max())
            breakout_up = entry_price > resistance
            breakout_down = entry_price < support
            volume_spike = is_volume_spike(data['df'])

            last_signal = get_last_signal(symbol, interval)
            trend_changed = (last_signal != data['trend_signal'])

            if (data['rsi_state'] is None
                and data['macd_cross'] is None
                and ma_cross is None
                and not breakout_up
                and not breakout_down
                and not trend_changed
                and not volume_spike):
                print(f"{symbol} ({interval}): No alert condition met.")
                continue

            # Compose all reasons
            alert_reasons = []
            if trend_changed:
                alert_reasons.append(f"Trend changed to <b>{data['trend_signal']}</b>")
            if data['macd_cross']:
                alert_reasons.append(f"MACD <b>{data['macd_cross'].capitalize()} crossover</b>")
            if data['rsi_state']:
                alert_reasons.append(f"RSI alert: <b>{data['rsi_state']}</b>")
            if ma_cross:
                alert_reasons.append(f"MA <b>{ma_cross.capitalize()} crossover</b>")
            if breakout_up:
                alert_reasons.append(f"🚀 <b>Price breakout above resistance!</b>")
            if breakout_down:
                alert_reasons.append(f"📉 <b>Price breakdown below support!</b>")
            if volume_spike:
                alert_reasons.append(f"🔊 <b>Volume Spike: {float(data['df']['volume'].iloc[-1]):,.2f} (>2x avg)</b>")
            alert_msg = "\n".join(alert_reasons) + "\n" if alert_reasons else ""

            tp1, tp2, tp3, tp4, stop_loss, sl_text, instructions = calculate_trade_levels(
                entry_price, data['trend_signal'], support, resistance)
            fibonacci_levels = calculate_fibonacci_levels(support, resistance, data['trend_signal'])

            # ---- Compose Message ----
            msg = (
                f"<b>{symbol} ({interval}) Trade Signal</b>\n\n"
                f"{alert_msg}"
                f"<b>Direction:</b> {data['trend_signal']}\n"
                f"<b>Entry (Price Point):</b> {entry_price:.2f}\n"
                f"<b>Support:</b> {support:.2f}\n"
                f"<b>Resistance:</b> {resistance:.2f}\n"
                f"<b>Fibonacci Levels:</b>\n"
            )
            for key, value in fibonacci_levels.items():
                msg += f"  {key}: {value:.2f}\n"
            msg += (
                f"\n<b>TP1:</b> {tp1:.2f}\n"
                f"<b>TP2:</b> {tp2:.2f}\n"
                f"<b>TP3:</b> {tp3:.2f}\n"
                f"<b>TP4:</b> {tp4:.2f}\n"
                f"<b>Stop Loss:</b> {sl_text}\n\n"
                f"<b>Instructions:</b> {instructions}\n\n"
                f"RSI (14): <b>{data['rsi']:.2f}</b> ({data['rsi_state'] if data['rsi_state'] else 'Neutral'})\n"
                f"Short MA (7): <b>{data['ma_short']:.2f}</b>\n"
                f"Long MA (25): <b>{data['ma_long']:.2f}</b>\n"
                f"MACD: <b>{data['macd_now']:.2f}</b>, MACD Signal: <b>{data['macd_signal_now']:.2f}</b>\n"
                f"Bollinger Bands: <b>{data['bb_upper_now']:.2f} / {data['bb_mid_now']:.2f} / {data['bb_lower_now']:.2f}</b>\n"
                f"Volume: <b>{float(data['df']['volume'].iloc[-1]):,.2f}</b>\n"
            )

            chart_path = save_chart(
                data['df'], data['trend_signal'], symbol, interval, data['rsi'],
                data['macd_line'], data['signal_line'],
                data['df']['close'].rolling(window=20).mean() + 2*data['df']['close'].rolling(window=20).std(),
                data['df']['close'].rolling(window=20).mean() - 2*data['df']['close'].rolling(window=20).std()
            )

            await send_signal_to_telegram(msg, chart_path=chart_path)
            set_last_alert_time(symbol, interval, now)
            log_signal(
                symbol, interval, data['trend_signal'], data['rsi'], data['macd_cross'], data['df']['close'].iloc[-1]
            )
            print(f"{symbol} ({interval}): Alert sent and logged.")

# ---- DUMMY WEB SERVER FOR CLOUD RUN ----
import threading
from flask import Flask

def run_dummy_server():
    app = Flask('dummy')
    @app.route("/")
    def index():
        return "Bot running!", 200
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# Start the dummy web server in a background thread
threading.Thread(target=run_dummy_server, daemon=True).start()

if __name__ == "__main__":
    asyncio.run(main())
