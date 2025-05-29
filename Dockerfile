# Start from the official Python image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the bot code
COPY . .

# Set environment variables (these get overridden in Cloud Run settings)
ENV TELEGRAM_TOKEN=""
ENV CHAT_ID=""

# Run the bot
CMD ["python", "main.py"]
