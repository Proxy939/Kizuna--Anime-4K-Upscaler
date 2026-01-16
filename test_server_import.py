"""
Test script to import api_server and see startup errors
"""
import sys
from pathlib import Path

# Mimic uvicorn environment
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=== Testing api_server import ===")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"sys.path[0]: {sys.path[0]}")
print()

try:
    print("Importing api.api_server...")
    from api import api_server
    print(f"✅ Import successful!")
    print(f"✅ AI_AVAILABLE = {api_server.AI_AVAILABLE}")
    if api_server.AI_AVAILABLE:
        print(f"✅ MODELS = {len(api_server.MODELS)}")
except Exception as e:
    print(f"❌ Import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
