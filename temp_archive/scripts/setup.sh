#!/bin/bash
# SPX Trading Bot Setup Script

echo "Setting up SPX Trading Bot..."

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file from template
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.template .env
    echo "Please edit .env file with your ThetaData credentials"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs data/raw data/processed

# Make main script executable
chmod +x main.py

echo ""
echo "Setup complete! Next steps:"
echo "1. Edit .env file with your ThetaData credentials"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Download data: python main.py download --days-back 30"
echo "4. Run backtest: python main.py backtest --strategy iron_condor --start-date 2024-01-01 --end-date 2024-12-31"
echo "5. Check status: python main.py status"