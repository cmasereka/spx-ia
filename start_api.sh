#!/bin/bash

echo "🚀 Starting SPX AI Trading Platform API..."
echo "======================================"

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "
try:
    import fastapi, uvicorn, websockets, pydantic, psutil
    print('✅ All required packages are installed')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Installing dependencies...')
    import subprocess
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
    print('✅ Dependencies installed')
"

echo ""
echo "🔍 Running quick API test..."
python -c "
from main import app
print('✅ FastAPI app loads successfully')
print('✅ Ready to start server')
"

if [ $? -ne 0 ]; then
    echo "❌ API test failed. Please check the error above."
    exit 1
fi

# Start the FastAPI server
echo ""
echo "✅ Starting server on http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔗 WebSocket endpoint: ws://localhost:8000/ws/{client_id}"
echo "🧪 Test script: python test_api.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload