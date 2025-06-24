#!/bin/bash

# NyxTrade Quick Start Script
# This script helps you get NyxTrade up and running quickly

set -e

echo "🚀 NyxTrade - Multi-Agent Cryptocurrency Trading AI"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run quality checks
run_quality_checks() {
    print_status "Running code quality checks..."

    if [ -f "scripts/quality_check.py" ]; then
        python scripts/quality_check.py
        if [ $? -eq 0 ]; then
            print_success "Quality checks passed"
        else
            print_warning "Some quality checks failed. Check quality_report.md for details."
        fi
    else
        print_warning "Quality check script not found"
    fi
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python $python_version is installed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Copy configuration files if they don't exist
    if [ ! -f "config/config.yaml" ]; then
        cp config/config.example.yaml config/config.yaml
        print_warning "Created config/config.yaml from example. Please edit it with your settings."
    fi
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Created .env from example. Please edit it with your API keys."
    fi
    
    # Create necessary directories
    mkdir -p logs data/cache monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    print_success "Environment setup completed"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created virtual environment"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    # Start database services
    docker-compose up -d postgres redis
    
    # Wait for services to be ready
    sleep 10
    
    # Initialize database schema
    source venv/bin/activate
    python scripts/init_db.py
    
    print_success "Database initialized"
}

# Start services
start_services() {
    print_status "Starting NyxTrade services..."
    
    # Start all services
    docker-compose up -d
    
    print_success "All services started"
    print_status "Services running:"
    echo "  - NyxTrade App: http://localhost:8000"
    echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an option:"
    echo "1) Full setup (recommended for first time)"
    echo "2) Start services only"
    echo "3) Stop services"
    echo "4) View logs"
    echo "5) Reset database"
    echo "6) Run quality checks"
    echo "7) Run tests"
    echo "8) Exit"
    echo ""
}

# Handle user choice
handle_choice() {
    case $1 in
        1)
            print_status "Starting full setup..."
            check_docker
            check_python
            setup_environment
            install_dependencies
            init_database
            start_services
            print_success "NyxTrade is now running!"
            ;;
        2)
            print_status "Starting services..."
            docker-compose up -d
            print_success "Services started"
            ;;
        3)
            print_status "Stopping services..."
            docker-compose down
            print_success "Services stopped"
            ;;
        4)
            print_status "Showing logs..."
            docker-compose logs -f nyxtrade
            ;;
        5)
            print_warning "This will reset all database data. Are you sure? (y/N)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                docker-compose down -v
                docker-compose up -d postgres redis
                sleep 10
                source venv/bin/activate
                python scripts/init_db.py
                print_success "Database reset completed"
            fi
            ;;
        6)
            print_status "Running quality checks..."
            run_quality_checks
            ;;
        7)
            print_status "Running tests..."
            if [ -d "venv" ]; then
                source venv/bin/activate
            fi
            python -m pytest tests/ -v
            ;;
        8)
            print_status "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
}

# Main execution
main() {
    # Check if running in interactive mode
    if [ -t 0 ]; then
        while true; do
            show_menu
            read -p "Enter your choice [1-6]: " choice
            handle_choice $choice
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Non-interactive mode - run full setup
        handle_choice 1
    fi
}

# Run main function
main "$@"
