"""Manual local query helper for the FastAPI chat endpoint.

Run the app first, then execute:
    python query_demo.py
"""

import requests


BASE_URL = "http://localhost:5000"


def main() -> None:
    session = requests.Session()
    session.get(f"{BASE_URL}/")
    response = session.post(
        f"{BASE_URL}/api/query",
        json={"question": "What is the capital of France?"},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)


if __name__ == "__main__":
    main()
