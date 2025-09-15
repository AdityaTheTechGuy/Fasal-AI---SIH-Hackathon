#!/bin/bash
# This script automates the setup and launch process for Fasal AI on macOS/Linux.

echo "--- Setting up Fasal AI environment ---"

# Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Compile language translations
echo "Compiling language files..."
pybabel compile -d translations

# Start the Flask server
echo "--- Starting Fasal AI server ---"
echo "Access the application at http://127.0.0.1:5000"
python app.py