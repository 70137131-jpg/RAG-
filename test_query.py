import requests

url = "http://localhost:5000/api/query"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer super-secret-student-key"
}
data = {
    "question": "What is the capital of France?"
}

try:
    response = requests.post(url, json=data, headers=headers)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
except Exception as e:
    print("Request failed:", e)
