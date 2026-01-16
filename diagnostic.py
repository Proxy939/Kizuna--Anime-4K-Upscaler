"""
Quick diagnostic script to identify server startup issues.
"""
import sys
from pathlib import Path

print("="*60)
print("KIZUNA SR - SERVER DIAGNOSTIC")
print("="*60)

# Test 1: Check Python version
print(f"\n[1] Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
print("   ✅ OK")

# Test 2: Check critical imports
print("\n[2] Testing Core Dependencies...")
try:
    from fastapi import FastAPI
    print("   ✅ FastAPI")
except ImportError as e:
    print(f"   ❌ FastAPI: {e}")

try:
    from PIL import Image
    print("   ✅ PIL/Pillow")
except ImportError as e:
    print(f"   ❌ PIL/Pillow: {e}")

try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ❌ PyTorch: {e}")

# Test 3: Check AI imports
print("\n[3] Testing AI Dependencies...")
try:
    from runtime.ai.model_registry import MODELS, get_default_model
    print(f"   ✅ Model Registry ({len(MODELS)} models)")
    default = get_default_model()
    if default:
        print(f"   ✅ Default Model: {default['label']}")
    else:
        print("   ⚠️  No default model set")
except ImportError as e:
    print(f"   ❌ Model Registry: {e}")
except Exception as e:
    print(f"   ❌ Model Registry Error: {e}")

try:
    from runtime.ai.ai_inference import get_device_info
    device_info = get_device_info()
    if device_info.get('cuda_available'):
        print(f"   ✅ GPU: {device_info.get('gpu_name', 'Unknown')}")
    else:
        print("   ⚠️  CPU Mode (slower)")
except ImportError as e:
    print(f"   ❌ AI Inference: {e}")
except Exception as e:
    print(f"   ⚠️  AI Inference Warning: {e}")

# Test 4: Check model file
print("\n[4] Checking Model File...")
model_path = Path("models/RealESRGAN_x4plus.pth")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Model file exists ({size_mb:.1f} MB)")
else:
    print(f"   ❌ Model file NOT FOUND: {model_path.absolute()}")

# Test 5: Check port availability
print("\n[5] Checking Port 8000...")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 8000))
sock.close()
if result == 0:
    print("   ⚠️  Port 8000 is IN USE (server might already be running)")
else:
    print("   ✅ Port 8000 is available")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf all tests pass, start the server with:")
print("  venv\\Scripts\\python.exe -m uvicorn api.api_server:app --host 127.0.0.1 --port 8000 --reload")
