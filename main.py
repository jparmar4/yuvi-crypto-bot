import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from telegram import Bot
from telegram.constants import ParseMode

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = Bot(token=TOKEN)

def fetch_data(symbol='BTCUSDT', interval='15m', limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    try:
        response = requests.get(url)
        data = response.json()
        if not data or len(data) < 25:
            return None
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def generate_signal(df):
    if df is None or len(df) < 25:
        # Not enough data to calculate moving averages
        return "NO SIGNAL (Insufficient Data)", None, None
    ma_short = df['close'].rolling(window=7).mean().iloc[-1]
    ma_long = df['close'].rolling(window=25).mean().iloc[-1]
    if ma_short > ma_long:
        return "LONG 🚀", ma_short, ma_long
    else:
        return "SHORT 📉", ma_short, ma_long

def save_chart(df, signal):
    plt.figure(figsize=(10,5))
    plt.plot(df['open_time'], df['close'], label='Close Price')
    plt.title(f"BTC/USDT - Signal: {signal}")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.grid(True)
    plt.legend()
    filename = f"chart.png"
    plt.savefig(filename)
    plt.close()
    return filename

def send_signal_to_telegram(signal, chart_path=None):
    bot.send_message(chat_id=CHAT_ID, text=f"Crypto Signal:\n{signal}", parse_mode=ParseMode.HTML)
    if chart_path:
        with open(chart_path, 'rb') as chart:
            bot.send_photo(chat_id=CHAT_ID, photo=chart)

if __name__ == "__main__":
    df = fetch_data()
    signal, short_ma, long_ma = generate_signal(df)
    if short_ma is None or long_ma is None:
        bot.send_message(chat_id=CHAT_ID, text="No signal: Not enough data to calculate moving averages.")
    else:
        chart_path = save_chart(df, signal)
        send_signal_to_telegram(signal=f"{signal}\nShort MA: {short_ma:.2f}\nLong MA: {long_ma:.2f}", chart_path=chart_path)
