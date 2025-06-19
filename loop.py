import time
import requests

url = "http://localhost:8001/sample"

while True:
    try:
        response = requests.get(url)
        print(f"[{time.strftime('%H:%M:%S')}] Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[{time.strftime('%H:%M:%S')}] Request failed: {e}")
    
    time.sleep(3)  # tunggu 3 detik (20x per menit = 60/20 = 3 detik)
