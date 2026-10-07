import sys
from pathlib import Path

# Add artifacts/api-server/python to sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))
