#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}== Medical AI Dashboard Setup Tool ==${NC}"
echo -e "${BLUE}======================================${NC}"

# Create and activate Python virtual environment
echo -e "\n${YELLOW}[1/4] Setting up Python virtual environment...${NC}"

# Check if venv module is available
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo -e "${RED}Error: Python venv module is not available.${NC}"
    echo -e "Please install it with: sudo apt-get install python3-venv"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo -e "\n${YELLOW}[2/4] Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Check if xgboost is installed properly
echo -e "\n${YELLOW}[3/4] Verifying XGBoost installation...${NC}"
if python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')" > /dev/null 2>&1; then
    echo -e "${GREEN}XGBoost installed successfully!${NC}"
else
    echo -e "${RED}XGBoost installation failed. Please check requirements.${NC}"
    exit 1
fi

# Start the Python API
echo -e "\n${YELLOW}[4/4] Starting services...${NC}"
echo "Starting Flask API..."
cd src/backend/api
python api.py &
API_PID=$!
echo -e "${GREEN}API started with PID: $API_PID${NC}"

# Navigate to frontend directory and install npm dependencies
echo "Installing Next.js dependencies..."
cd ../../frontend
npm install

# Start Next.js frontend
echo "Starting Next.js frontend..."
echo -e "${GREEN}Dashboard will be available at: http://localhost:3000${NC}"
echo -e "${GREEN}API will be available at: http://localhost:5000${NC}"
echo -e "\n${BLUE}Press Ctrl+C to stop both services${NC}\n"

# Define cleanup function for handling interrupt
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    kill $API_PID
    echo "API stopped"
    exit 0
}

# Register the cleanup function for SIGINT
trap cleanup SIGINT

# Start the frontend
npm run dev

# If npm run dev exits, also stop the API
cleanup