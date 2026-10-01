import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from app.services.dataset_inventory import DatasetInventory
from app.routes.multi_registration import router
from fastapi.testclient import TestClient
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def chk(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} - {name}")

def run_api_tests():
    print("\nRunning Phase 11 API Tests")
    
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "ch2_ohrc_01.xml").write_text("<test></test>", encoding="utf-8")
        (root / "ch2_ohrc_01.img").write_bytes(b"data")
        
        response = client.get(f"/dataset-inventory?root_path={td}")
        chk("N. API serialization (discovery)", response.status_code == 200)
        if response.status_code == 200:
            data = response.json()
            chk("N2. API returns summary and products", 'summary' in data and 'products' in data)
            chk("N3. Discovered files match", data['summary']['total_discovered'] == 2)
            chk("N4. Product identity safe serialization", len(data['products']) == 1)

if __name__ == "__main__":
    run_api_tests()
