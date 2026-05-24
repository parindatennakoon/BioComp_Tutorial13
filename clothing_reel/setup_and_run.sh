#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Clothing Reel Generator Setup ==="

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install from https://python.org"
  exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q "moviepy==1.0.3" imageio[ffmpeg]
pip install -q -r requirements.txt

# Install Playwright browser
echo "Installing Playwright browser..."
playwright install chromium

# API key setup
if [ ! -f ".env" ] || ! grep -q "sk-ant" .env 2>/dev/null; then
  echo ""
  echo "Paste your Anthropic API key (from console.anthropic.com):"
  read -r apikey
  echo "ANTHROPIC_API_KEY=$apikey" > .env
  echo "API key saved."
fi

# Run
echo ""
echo "Starting server at http://127.0.0.1:5000"
echo "Press CTRL+C to stop."
echo ""
python app.py
