import requests
import random

URL = "http://127.0.0.1:8080/estimate"

def send(data):
    requests.post(URL, json=data)

print("Normal traffic...")
for _ in range(35):
    send({
        "temperature_c": random.uniform(20, 35),
        "building_sqm": random.uniform(100, 400),
        "occupancy_pct": random.uniform(30, 80),
        "is_weekday": random.randint(0, 1)
    })

print("Drift traffic...")
for _ in range(15):
    send({
        "temperature_c": random.uniform(50, 90),
        "building_sqm": random.uniform(200, 500),
        "occupancy_pct": random.uniform(120, 200),
        "is_weekday": random.randint(0, 1)
    })

print("Done sending 50 requests")
