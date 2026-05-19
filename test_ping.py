import requests
import time
import sys

bridge_url = "http://10.0.0.85:11437"

print("Waiting for bridge to report backend is healthy...")
max_retries = 60
for i in range(max_retries):
    try:
        health_res = requests.get(f"{bridge_url}/health")
        if health_res.status_code == 200:
            print("\\nBackend is ready! Testing inference...")
            break
    except Exception:
        pass
    sys.stdout.write(".")
    sys.stdout.flush()
    time.sleep(5)
else:
    print("\\nTimeout waiting for backend.")
    sys.exit(1)

try:
    r = requests.post(f"{bridge_url}/api/chat", json={
        "model": "qwen2.5-coder-7b", 
        "messages": [{"role": "user", "content": "Write a short Python function to calculate fibonacci."}], 
        "stream": False
    })
    print(f"\\nResponse: {r.status_code}")
    print(r.json())
    sys.exit(0)
except Exception as e:
    print(f"\\nFailed to connect: {e}")
    sys.exit(1)
