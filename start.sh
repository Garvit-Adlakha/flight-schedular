#!/bin/bash

# Flight Schedule Optimization System - Comprehensive Startup Script
# This script sets up and starts the entire system in the proper order

echo "🚀 Starting Flight Schedule Optimization System..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i:$1 >/dev/null 2>&1
}

# Function to check if a service is running
service_running() {
    local service=$1
    if command_exists systemctl; then
        systemctl is-active --quiet $service
    elif command_exists brew; then
        brew services list | grep -q "$service.*started"
    else
        pgrep -f $service >/dev/null
    fi
}

# Function to check local PostgreSQL installation
check_local_postgres() {
    if command_exists psql; then
        echo -e "${GREEN}✅ PostgreSQL found${NC}"
        
        # Try to connect to check if it's running
        if pg_isready >/dev/null 2>&1; then
            echo -e "${GREEN}✅ PostgreSQL is running${NC}"
        else
            echo -e "${YELLOW}⚠️  PostgreSQL is not running. Attempting to start...${NC}"
            if command_exists systemctl; then
                sudo systemctl start postgresql 2>/dev/null || echo -e "${YELLOW}⚠️  Could not start PostgreSQL automatically${NC}"
            elif command_exists brew; then
                brew services start postgresql 2>/dev/null || echo -e "${YELLOW}⚠️  Could not start PostgreSQL automatically${NC}"
            else
                echo -e "${YELLOW}⚠️  Please start PostgreSQL manually${NC}"
            fi
        fi
    else
        echo -e "${RED}❌ PostgreSQL not found locally. Please install PostgreSQL or use Docker.${NC}"
        exit 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Killing existing process on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null
        sleep 2
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=1
    
    echo -e "${BLUE}⏳ Waiting for $service_name to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to start within expected time${NC}"
    return 1
}

# Function to install Ollama
install_ollama() {
    echo -e "${BLUE}🤖 Setting up Ollama...${NC}"
    
    if command_exists ollama; then
        echo -e "${GREEN}✅ Ollama already installed${NC}"
    else
        echo -e "${YELLOW}📥 Installing Ollama...${NC}"
        
        # Install Ollama based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -fsSL https://ollama.ai/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install ollama
            else
                echo -e "${RED}❌ Please install Homebrew first: https://brew.sh/${NC}"
                return 1
            fi
        else
            echo -e "${RED}❌ Unsupported OS. Please install Ollama manually: https://ollama.ai/${NC}"
            return 1
        fi
    fi
    
    # Start Ollama service
    if ! service_running ollama; then
        echo -e "${BLUE}🔄 Starting Ollama service...${NC}"
        if command_exists systemctl; then
            sudo systemctl start ollama
        elif command_exists brew; then
            brew services start ollama
        else
            ollama serve &
        fi
        
        # Wait for Ollama to start
        wait_for_service "Ollama" "localhost" "11434" 30
        if [ $? -ne 0 ]; then
            return 1
        fi
    else
        echo -e "${GREEN}✅ Ollama service is running${NC}"
    fi
    
    # Check if model is available
    echo -e "${BLUE}🔍 Checking Ollama model...${NC}"
    if ollama list | grep -q "llama3.2"; then
        echo -e "${GREEN}✅ Llama3.2 model is available${NC}"
    else
        echo -e "${YELLOW}📥 Pulling Llama3.2 model (this may take a while)...${NC}"
        ollama pull llama3.2
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Llama3.2 model downloaded successfully${NC}"
        else
            echo -e "${RED}❌ Failed to download Llama3.2 model${NC}"
            return 1
        fi
    fi
    
    return 0
}

# Function to setup database
setup_database() {
    echo -e "${BLUE}🗄️  Setting up database...${NC}"
    
    cd ml-model
    
    # Check if database setup script exists
    if [ ! -f "database/setup.py" ]; then
        echo -e "${RED}❌ Database setup script not found${NC}"
        return 1
    fi
    
    # Run database setup
    echo -e "${BLUE}🔧 Running database setup...${NC}"
    python3 database/setup.py 2>/dev/null || python database/setup.py 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Database setup completed${NC}"
    else
        echo -e "${YELLOW}⚠️  Database setup had some issues, but continuing...${NC}"
    fi
    
    cd ..
    return 0
}

# Main startup sequence
main() {
    # Check prerequisites
    echo -e "${BLUE}📋 Checking prerequisites...${NC}"
    
    if ! command_exists python3 && ! command_exists python; then
        echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    if ! command_exists node; then
        echo -e "${RED}❌ Node.js not found. Please install Node.js 16+${NC}"
        exit 1
    fi
    
    if ! command_exists npm; then
        echo -e "${RED}❌ npm not found. Please install npm${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
    
    # Step 1: Setup PostgreSQL
    echo -e "${PURPLE}🐘 Step 1: Setting up PostgreSQL...${NC}"
    
    # Check if local PostgreSQL is already running
    if pg_isready >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Local PostgreSQL is already running${NC}"
        echo -e "${BLUE}ℹ️  Using local PostgreSQL installation${NC}"
    elif command_exists docker && docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Docker is available${NC}"
        
        if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
            echo -e "${GREEN}✅ docker-compose is available${NC}"
            
            # Check if port 5432 is free
            if ! lsof -i:5432 >/dev/null 2>&1; then
                echo -e "${BLUE}🐳 Port 5432 is free, starting PostgreSQL in Docker...${NC}"
                
                # Stop any existing containers
                docker-compose down postgres 2>/dev/null || docker compose down postgres 2>/dev/null
                
                # Start PostgreSQL
                if docker-compose up -d postgres 2>/dev/null || docker compose up -d postgres 2>/dev/null; then
                    echo -e "${GREEN}✅ PostgreSQL started successfully in Docker${NC}"
                    
                    # Wait for PostgreSQL to be ready
                    wait_for_service "PostgreSQL" "localhost" "5432" 30
                    if [ $? -ne 0 ]; then
                        exit 1
                    fi
                else
                    echo -e "${RED}❌ Failed to start PostgreSQL in Docker, falling back to local...${NC}"
                    check_local_postgres
                fi
            else
                echo -e "${YELLOW}⚠️  Port 5432 is in use, using local PostgreSQL...${NC}"
                check_local_postgres
            fi
        else
            echo -e "${YELLOW}⚠️  docker-compose not found. Checking local PostgreSQL...${NC}"
            check_local_postgres
        fi
    else
        echo -e "${YELLOW}⚠️  Docker not available. Checking local PostgreSQL...${NC}"
        check_local_postgres
    fi
    
    # Step 2: Install Python dependencies
    echo -e "${PURPLE}🐍 Step 2: Installing Python dependencies...${NC}"
    
    if [ ! -d "ml-model" ]; then
        echo -e "${RED}❌ ml-model directory not found${NC}"
        exit 1
    fi
    
    cd ml-model
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}❌ requirements.txt not found in ml-model directory${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}📦 Installing Python packages...${NC}"
    python3 -m pip install -r requirements.txt --quiet 2>/dev/null || python -m pip install -r requirements.txt --quiet 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Python dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️  Some Python dependencies may need manual installation${NC}"
    fi
    
    cd ..
    
    # Step 3: Install Node.js dependencies
    echo -e "${PURPLE}🟢 Step 3: Installing Node.js dependencies...${NC}"
    
    if [ ! -d "frontend" ]; then
        echo -e "${RED}❌ frontend directory not found${NC}"
        exit 1
    fi
    
    cd frontend
    if [ ! -f "package.json" ]; then
        echo -e "${RED}❌ package.json not found in frontend directory${NC}"
        exit 1
    fi
    
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}📦 Installing Node.js packages...${NC}"
        npm install --silent
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
        else
            echo -e "${RED}❌ Failed to install Node.js dependencies${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Node.js dependencies already installed${NC}"
    fi
    
    cd ..
    
    # Step 4: Setup Ollama
    echo -e "${PURPLE}🤖 Step 4: Setting up Ollama...${NC}"
    
    if ! install_ollama; then
        echo -e "${YELLOW}⚠️  Ollama setup failed, but continuing without LLM features...${NC}"
    fi
    
    # Step 5: Setup database (skip if already configured)
    echo -e "${PURPLE}🗄️  Step 5: Checking database connection...${NC}"
    
    # Test database connection
    if PGPASSWORD=flight_password psql -h localhost -U flight_user -d flight_schedule_optimization -c "SELECT 1;" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Database connection successful${NC}"
    else
        echo -e "${YELLOW}⚠️  Database connection failed, attempting setup...${NC}"
        if ! setup_database; then
            echo -e "${YELLOW}⚠️  Database setup had issues, but continuing...${NC}"
        fi
    fi
    
    # Step 6: Clean up any existing processes
    echo -e "${PURPLE}🧹 Step 6: Cleaning up existing processes...${NC}"
    
    kill_port 8000  # Backend API port
    kill_port 3000  # Frontend port
    
    # Create log directory
    mkdir -p logs
    
    # Step 7: Start backend API
    echo -e "${PURPLE}🔄 Step 7: Starting Backend API...${NC}"
    
    cd ml-model
    nohup python3 api/start_api.py > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    wait_for_service "Backend API" "localhost" "8000" 30
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Backend API failed to start. Check logs/backend.log${NC}"
        cat logs/backend.log
        exit 1
    fi
    
    echo -e "${GREEN}✅ Backend API started successfully on http://localhost:8000${NC}"
    
    # Step 8: Start frontend
    echo -e "${PURPLE}🔄 Step 8: Starting Frontend...${NC}"
    
    cd frontend
    
    # Set environment variable for API URL
    export REACT_APP_API_URL=http://localhost:8000
    
    # Start React development server
    nohup npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    wait_for_service "Frontend" "localhost" "3000" 30
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Frontend may still be starting. Check logs/frontend.log${NC}"
    else
        echo -e "${GREEN}✅ Frontend started successfully on http://localhost:3000${NC}"
    fi
    
    # Final status
    echo ""
    echo -e "${GREEN}🎉 Flight Schedule Optimization System Started Successfully!${NC}"
    echo "=================================================="
    echo -e "${CYAN}📊 Frontend Dashboard:${NC} http://localhost:3000"
    echo -e "${CYAN}🔧 Backend API:${NC}       http://localhost:8000"
    echo -e "${CYAN}📖 API Documentation:${NC} http://localhost:8000/docs"
    echo -e "${CYAN}🤖 Ollama Status:${NC}     $(ollama list | grep -q "llama3.2" && echo "✅ Active" || echo "❌ Inactive")"
    echo ""
    echo -e "${YELLOW}💡 Usage:${NC}"
    echo "  1. Open http://localhost:3000 in your browser"
    echo "  2. Upload flight data using the 'Data Upload' tab"
    echo "  3. Explore analytics in the 'Analytics Dashboard'"
    echo "  4. Chat with AI in the 'AI Assistant' tab"
    echo ""
    echo -e "${YELLOW}🔍 Process Information:${NC}"
    echo "  Backend PID: $BACKEND_PID"
    echo "  Frontend PID: $FRONTEND_PID"
    echo ""
    echo -e "${YELLOW}📝 Logs:${NC}"
    echo "  Backend:  logs/backend.log"
    echo "  Frontend: logs/frontend.log"
    echo ""
    echo -e "${YELLOW}🛑 To stop the system:${NC}"
    echo "  ./stop.sh"
    echo "  or"
    echo "  kill $BACKEND_PID $FRONTEND_PID"
    echo ""
    
    # Save PIDs for stop script
    echo "$BACKEND_PID" > logs/backend.pid
    echo "$FRONTEND_PID" > logs/frontend.pid
    
    echo -e "${GREEN}✅ System is ready! Enjoy optimizing flight schedules! ✈️${NC}"
}

# Run main function
main "$@"
