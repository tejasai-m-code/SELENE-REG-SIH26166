from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)
path = "../test_rgba.png"
with open(path, "rb") as f:
    files = {
        "source": ("test_rgba.png", f, "image/png"),
        "reference": ("test_rgba.png", open(path, "rb"), "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "sift"})
    print(response.status_code)
    print(response.json())
