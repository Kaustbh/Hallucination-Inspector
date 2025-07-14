#!/usr/bin/env python3
"""
Hallucination Detector Launcher
Checks environment setup and launches the Streamlit app
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Some environment variables are not set:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 You can still use the app by entering API keys in the sidebar!")
        print("📝 Or set up your .env file:")
        print("   1. Create a .env file in this directory")
        print("   2. Add your API keys:")
        print("      GROQ_API_KEY=your_groq_key_here")
        print("      GOOGLE_API_KEY=your_google_key_here")
        print("   3. Restart the application")
        return False
    
    print("✅ Environment variables are properly configured")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import uqlm
        import langchain
        import plotly
        import pandas
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Please install dependencies:")
        print("   uv sync")
        return False

def main():
    """Main launcher function"""
    print("🤖 Hallucination Detector Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check environment (but don't exit if missing)
    check_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting Hallucination Detector...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("🔑 You can enter API keys directly in the app sidebar")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()