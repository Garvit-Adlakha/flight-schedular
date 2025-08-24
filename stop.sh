#!/bin/bash

# Flight Schedule Optimization System - Stop Script
# This script stops both the backend API and frontend React app

echo "🛑 Stopping Flight Schedule Optimization System..."
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to kill process on port
kill_port() {
    local port=$1
    local service_name=$2
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Stopping $service_name on port $port (PID: $pid)...${NC}"
        kill -TERM $pid 2>/dev/null
        sleep 3
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Force killing $service_name...${NC}"
            kill -9 $pid 2>/dev/null
        fi
        
        if ! kill -0 $pid 2>/dev/null; then
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${RED}❌ Failed to stop $service_name${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  $service_name not running on port $port${NC}"
    fi
}

# Stop using saved PIDs if available
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    echo -e "${YELLOW}Stopping backend (PID: $BACKEND_PID)...${NC}"
    kill -TERM $BACKEND_PID 2>/dev/null
    sleep 2
    kill -9 $BACKEND_PID 2>/dev/null
    rm -f logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    echo -e "${YELLOW}Stopping frontend (PID: $FRONTEND_PID)...${NC}"
    kill -TERM $FRONTEND_PID 2>/dev/null
    sleep 2
    kill -9 $FRONTEND_PID 2>/dev/null
    rm -f logs/frontend.pid
fi

# Stop by port numbers (fallback)
kill_port 8000 "Backend API"
kill_port 3000 "Frontend"

# Kill any remaining React/Node processes
echo -e "${BLUE}🧹 Cleaning up remaining processes...${NC}"
pkill -f "react-scripts" 2>/dev/null
pkill -f "start_api.py" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null

# Stop Docker containers if running
echo -e "${BLUE}🐳 Stopping Docker containers...${NC}"
if docker-compose down 2>/dev/null || docker compose down 2>/dev/null; then
    echo -e "${GREEN}✅ Docker containers stopped${NC}"
else
    echo -e "${BLUE}ℹ️  No Docker containers running${NC}"
fi

# Remove any dangling containers
dangling_containers=$(docker ps -aq --filter "name=flight-*" 2>/dev/null)
if [ ! -z "$dangling_containers" ]; then
    echo -e "${BLUE}🧹 Removing dangling containers...${NC}"
    docker rm -f $dangling_containers 2>/dev/null
fi

# Clean up log files (optional)
read -p "$(echo -e ${YELLOW}"Do you want to clear log files? (y/N): "${NC})" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f logs/backend.log logs/frontend.log
    echo -e "${GREEN}✅ Log files cleared${NC}"
fi

echo ""
echo -e "${GREEN}✅ Flight Schedule Optimization System stopped successfully!${NC}"
echo ""
echo -e "${BLUE}💡 To start again, run:${NC} ./start.sh"
echo ""
