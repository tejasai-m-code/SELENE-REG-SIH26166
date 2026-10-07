import sys
from pathlib import Path
sys.path.insert(0, str(Path("artifacts/api-server/python").resolve()))

import json
from app.services.hardware_manager import get_hardware_status

status = get_hardware_status()
print(json.dumps(status, indent=2))
