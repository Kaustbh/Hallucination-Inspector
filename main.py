# def main():
#     print("Hello from hallucination-inspector!")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Test script to check environment variable loading
"""

import os
from dotenv import load_dotenv

print("🔍 Testing Environment Variable Loading")
print("=" * 40)

# Check if .env file exists
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ .env file found: {env_file}")
else:
    print(f"❌ .env file not found: {env_file}")
    print("   Please create a .env file with your API keys")

# Try to load environment variables
print("\n📥 Loading environment variables...")
load_dotenv()

# Check required variables
required_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY']
missing_vars = []

for var in required_vars:
    value = os.getenv(var)
    if value:
        # Show first few characters for security
        masked_value = value[:8] + "..." if len(value) > 8 else "***"
        print(f"✅ {var}: {masked_value}")
    else:
        print(f"❌ {var}: Not found")
        missing_vars.append(var)

if missing_vars:
    print(f"\n❌ Missing variables: {', '.join(missing_vars)}")
    print("\n📝 To fix this:")
    print("1. Create a .env file in your project directory")
    print("2. Add your API keys like this:")
    print("   GROQ_API_KEY=your_actual_groq_key")
    print("   GOOGLE_API_KEY=your_actual_google_key")
else:
    print("\n✅ All environment variables are set!")
    print("🚀 You can now run: uv run run.py")