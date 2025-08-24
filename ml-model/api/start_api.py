#!/usr/bin/env python3
"""
API Startup Script
Starts the Flight Schedule Optimization API server
"""

import uvicorn
import sys
import os
from pathlib import Path

def start_api_server():
    """Start the API server with proper configuration"""
    
    print("🚀 Starting Flight Schedule Optimization API Server...")
    print("=" * 60)
    
    # Set up paths
    api_dir = Path(__file__).parent
    os.chdir(api_dir)
    
    print(f"📂 Working directory: {api_dir}")
    print(f"🌐 API will be available at: http://localhost:8000")
    print(f"📚 Interactive docs at: http://localhost:8000/docs")
    print(f"📋 ReDoc docs at: http://localhost:8000/redoc")
    
    print("\n🔧 Server Configuration:")
    print("   - Host: 0.0.0.0 (all interfaces)")
    print("   - Port: 8000")
    print("   - Reload: True (development mode)")
    print("   - Log level: info")
    
    print("\n⚡ Starting server...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    start_api_server()
