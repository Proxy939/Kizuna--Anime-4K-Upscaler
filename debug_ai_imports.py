"""
Debug script to capture exact import error at server startup
"""
import sys
import traceback

print("=== AI Import Debug ===")
print("Python:", sys.version)
print("CWD:", sys.path[0])
print()

try:
    print("Attempting: from runtime.ai.model_registry import MODELS")
    from runtime.ai.model_registry import MODELS, get_default_model, get_model
    print("✅ model_registry OK")
    print(f"✅ MODELS count: {len(MODELS)}")
except Exception as e:
    print(f"❌ model_registry FAILED: {type(e).__name__}")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\nAttempting: from runtime.ai.ai_inference import get_device_info")
    from runtime.ai.ai_inference import get_device_info
    print("✅ ai_inference OK")
    info = get_device_info()
    print(f"✅ Device: {info}")
except Exception as e:
    print(f"❌ ai_inference FAILED: {type(e).__name__}")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL IMPORTS SUCCESSFUL ===")
