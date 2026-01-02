#!/bin/bash
# Setup script - keeps your global env clothed

set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating venv..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! Run the scanner with:"
echo "  source venv/bin/activate"
echo "  python collar_scan.py"
