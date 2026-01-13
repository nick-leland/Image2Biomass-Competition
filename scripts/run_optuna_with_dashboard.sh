#!/bin/bash
# Launch Optuna optimization with live dashboard monitoring
# Usage: bash scripts/run_optuna_with_dashboard.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Optuna Optimization with Live Dashboard${NC}"
echo -e "${GREEN}============================================${NC}"

# Activate virtual environment
source .venv/bin/activate

# Parse arguments
N_TRIALS=${1:-100}
PORT=${2:-8080}
STUDY_NAME="advanced_models_$(date +%Y%m%d_%H%M%S)"

echo -e "\n${BLUE}Configuration:${NC}"
echo "  Study name: $STUDY_NAME"
echo "  Number of trials: $N_TRIALS"
echo "  Dashboard port: $PORT"

# Create storage path
STORAGE="sqlite:///experiments/optuna_studies/${STUDY_NAME}.db"

echo -e "\n${YELLOW}Step 1: Starting Optuna Dashboard${NC}"
echo "Dashboard will be available at: http://localhost:$PORT"

# Start dashboard in background
nohup optuna-dashboard $STORAGE --port $PORT > experiments/optuna_studies/${STUDY_NAME}_dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "Dashboard PID: $DASHBOARD_PID"
sleep 2

# Check if dashboard started successfully
if ps -p $DASHBOARD_PID > /dev/null; then
    echo -e "${GREEN}Dashboard started successfully!${NC}"
    echo -e "${GREEN}Open in browser: http://localhost:$PORT${NC}"
else
    echo -e "${YELLOW}Warning: Dashboard may not have started correctly${NC}"
fi

echo -e "\n${YELLOW}Step 2: Starting Optuna Optimization${NC}"
echo "This will run $N_TRIALS trials (may take 12-24 hours)"
echo ""

# Run optimization
python scripts/optimize_hyperparams.py \
    --n_trials $N_TRIALS \
    --study_name $STUDY_NAME \
    --storage $STORAGE \
    --dashboard_port $PORT

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Optimization Complete!${NC}"
echo -e "${GREEN}============================================${NC}"

echo -e "\n${BLUE}Cleaning up dashboard process...${NC}"
kill $DASHBOARD_PID 2>/dev/null || echo "Dashboard process already terminated"

echo -e "\n${GREEN}You can restart the dashboard anytime with:${NC}"
echo "  optuna-dashboard $STORAGE --port $PORT"
