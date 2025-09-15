#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# This script automates the setup and launch process for Fasal AI on macOS/Linux.

echo "--- Setting up Fasal AI environment ---"

# Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Activated."

# Install/update dependencies
echo "Installing dependencies from requirements.txt..."
# Use python from venv to ensure we use the correct pip
python3 -m pip install -r requirements.txt

# Compile language translations
echo "Compiling language files..."
pybabel compile -d translations

# Start the Flask server
echo "--- Starting Fasal AI server ---"
echo "Access the application at http://1227.0.0.1:5000"
python3 app.py

