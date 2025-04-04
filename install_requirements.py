import subprocess
import sys
import importlib

# List of required modules
modules = [
    "autocorrect",
    "opencv",
    "streamlit",
    "pytesseract",
    "opencv-python",
    "numpy",
    "pillow",
    "PyQt5",
    "python-docx",
    "fastapi",
    "python-multipart",
    "transformers",
    "datasets",
    "torch",
    "sentencepiece"
]

def install_module(module):
    """Check if a module is installed, and install it if not."""
    try:
        importlib.import_module(module)
        print(f"{module} is already installed.")
    except ImportError:
        print(f"Installing {module}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", module], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully installed {module}")
        else:
            print(f"❌ Failed to install {module}. Error:\n{result.stderr}")

# Install required modules
for module in modules:
    install_module(module)

print("✅ All required modules are installed.")
