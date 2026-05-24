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
pip install -q imageio[ffmpeg]
pip install -q -r requirements.txt

# Install Playwright browser
echo "Installing Playwright browser..."
playwright install chromium

# Anthropic API key
if [ ! -f ".env" ] || ! grep -q "sk-ant" .env 2>/dev/null; then
  echo ""
  echo "Enter your Anthropic API key (from console.anthropic.com):"
  read -r apikey
  echo "ANTHROPIC_API_KEY=$apikey" > .env
  echo "Anthropic key saved."
fi

# Replicate API token (optional, for AI style vibes)
if ! grep -q "REPLICATE_API_TOKEN" .env 2>/dev/null; then
  echo ""
  echo "Enter your Replicate API token for AI style vibes"
  echo "(from replicate.com — press Enter to skip, you can add it later):"
  read -r reptoken
  if [ -n "$reptoken" ]; then
    echo "REPLICATE_API_TOKEN=$reptoken" >> .env
    echo "Replicate token saved."
  else
    echo "Skipped. Add REPLICATE_API_TOKEN to .env later to enable AI vibes."
  fi
fi

# Run
echo ""
echo "Starting server at http://127.0.0.1:5000"
echo "Press CTRL+C to stop."
echo ""
python app.py
