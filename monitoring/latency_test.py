import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


API_URL = "https://breast-cancer-api-1037614303083.asia-south1.run.app/predict"

payload = {
    "features": [
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
}


TOTAL_REQUESTS = 100
CONCURRENT_REQUESTS = 10


def send_request():
    start_time = time.perf_counter()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    end_time = time.perf_counter()

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status code {response.status_code}"
        )

    return (end_time - start_time) * 1000


print("Cloud Run Latency Test")
print("=" * 60)
print(f"Total requests: {TOTAL_REQUESTS}")
print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
print("Starting load test...\n")


latencies = []

with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:

    futures = [
        executor.submit(send_request)
        for _ in range(TOTAL_REQUESTS)
    ]

    for future in as_completed(futures):
        latencies.append(future.result())


latencies.sort()

p50 = statistics.median(latencies)

p95_index = int(0.95 * len(latencies)) - 1
p95 = latencies[p95_index]

minimum = min(latencies)
maximum = max(latencies)
average = statistics.mean(latencies)


print("=" * 60)
print("Latency Results")
print("=" * 60)

print(f"Successful requests : {len(latencies)}")
print(f"Minimum latency     : {minimum:.2f} ms")
print(f"Average latency     : {average:.2f} ms")
print(f"P50 latency         : {p50:.2f} ms")
print(f"P95 latency         : {p95:.2f} ms")
print(f"Maximum latency     : {maximum:.2f} ms")
