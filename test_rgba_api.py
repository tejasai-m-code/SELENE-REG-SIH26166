import httpx
import asyncio

async def test_api():
    async with httpx.AsyncClient() as client:
        with open("test_rgba.png", "rb") as f:
            files = {
                "source": ("test_rgba.png", f, "image/png"),
                "reference": ("test_rgba.png", open("test_rgba.png", "rb"), "image/png")
            }
            response = await client.post("http://127.0.0.1:8000/api/register", files=files, data={"detector": "sift"})
            print(response.status_code)
            print(response.text)

asyncio.run(test_api())
