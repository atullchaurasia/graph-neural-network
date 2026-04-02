import requests
import time
import csv
import os
from datetime import datetime

"""
TOMTOM TRAFFIC DATA SCRAPER FOR INDORE
======================================
This script grabs live traffic speeds from Indore roads to build 
your machine learning dataset!

STEP 1: Go to TomTom Developer Portal, create a Free Account.
STEP 2: Get your API Key and paste it below.
STEP 3: Set this script to run every 5-15 minutes using Windows Task Scheduler.
        It will automatically append new speeds to the CSV for training!
"""

# 🛑 PASTE YOUR API KEY HERE 🛑
API_KEY = "7HikJfgeIcLx5U8PJzDhp6zGaigJT5Ne"

# Add the GPS coordinates of intersections you want to monitor here (like Loop Detectors)
INDORE_SENSORS = [
    {"id": "sensor_001", "name": "Rajwada Palace", "lat": 22.7184, "lon": 75.8547},
    {"id": "sensor_002", "name": "Bhawarkua Square", "lat": 22.6928, "lon": 75.8749},
    {"id": "sensor_003", "name": "Palasia Square",   "lat": 22.7233, "lon": 75.8844},
    {"id": "sensor_004", "name": "Indore Airport",   "lat": 22.7203, "lon": 75.8032},
    {"id": "sensor_005", "name": "Musakhedi Square", "lat": 22.6928, "lon": 75.8997},
    # You can add as many as you want here!
]

CSV_FILE = os.path.join("data", "indore_live_traffic.csv")

def fetch_speed(lat, lon):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": API_KEY,
        "point": f"{lat},{lon}",
        "unit": "KMPH"
    }
    res = requests.get(url, params=params)
    
    if res.status_code == 200:
        data = res.json()["flowSegmentData"]
        # Gives exactly how fast cars are going, and what the speed limit is!
        return data["currentSpeed"], data["freeFlowSpeed"]
    elif res.status_code == 403:
        print("Error 403: Invalid API Key!")
        return None, None
    elif res.status_code == 429:
        print("Error 429: Rate Limit Exceeded!")
        return None, None
    else:
        print(f"Error {res.status_code} fetching {lat},{lon}")
        # Sometimes a coordinate might be off a road, returning 404
        return None, None

def main():
    if API_KEY == "YOUR_TOMTOM_API_KEY_HERE":
        print("❌ STOP! You must paste your TomTom API Key into line 17 first!")
        return

    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    
    # Open CSV in "append" mode so it builds an infinite timeline of history
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if the file is brand new
        if not file_exists:
            writer.writerow(["timestamp", "sensor_id", "name", "current_speed_kmh", "free_flow_speed_kmh"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Pinging TomTom API for {len(INDORE_SENSORS)} sensors...")
        
        # Ping the API for every location in our list
        for sensor in INDORE_SENSORS:
            curr_speed, free_speed = fetch_speed(sensor['lat'], sensor['lon'])
            
            if curr_speed is not None:
                writer.writerow([timestamp, sensor['id'], sensor['name'], curr_speed, free_speed])
                print(f"   [SUCCESS] {sensor['name']:<20} -> Traffic: {curr_speed} km/h (Limit: {free_speed} km/h)")
                
            # Sleep 0.2s so TomTom doesn't block us for spamming too fast
            time.sleep(0.2)
            
    print(f"Data saved cleanly to {CSV_FILE}.")

if __name__ == "__main__":
    main()
