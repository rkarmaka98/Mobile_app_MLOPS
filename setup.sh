#!/bin/bash

# Create Python virtual environment
python3.10 -m venv venv

# Activate Python virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install npm dependencies
npm install

echo "Setup complete! To start the development environment:"
echo "1. Start the Python server: source venv/bin/activate && python server.py"
echo "2. Start the React Native app: npm start" 