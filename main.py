import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import heapq
import random
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

pune_center = (18.5204, 73.8567)  
max_station_queue = 5
min_soc_thres = 10 
soc_consumption_rate = 4  
swap_time = 4  

def generate_random_coordinates(center, radius_km, count):
    coords = []
    for _ in range(count):
        radius_deg = radius_km / 111.32
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, radius_deg)
        lat = center[0] + distance * np.sin(angle)
        lng = center[1] + distance * np.cos(angle)
        coords.append((lat, lng))
    return coords

def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  
    return distance

def estimate_travel_time_minutes(distance_km):
    avg_speed_kmh = 20
    return (distance_km / avg_speed_kmh) * 60

def generate_mock_data(num_riders=100, num_stations=3):
    now = datetime.now()
    station_locations = generate_random_coordinates(pune_center, 5, num_stations)
    stations = []
    for i, (lat, lng) in enumerate(station_locations):
        stations.append({
            "station_id": f"S_{chr(65+i)}",
            "lat": lat,
            "lng": lng,
            "queue_len": random.randint(0, 3)  
        })
    
    rider_locations = generate_random_coordinates(pune_center, 10, num_riders)
    riders = []
    
    for i, (lat, lng) in enumerate(rider_locations):
        soc_pct = random.randint(10, 85)
        status = "on_gig" if random.random() < 0.7 else "idle"
        
        rider = {
            "rider_id": f"R{i:03d}",
            "lat": lat,
            "lng": lng,
            "soc_pct": soc_pct,
            "status": status
        }
        
        if status == "on_gig":
            km_to_finish = round(random.uniform(0.5, 8.0), 1)
            minutes_to_finish = estimate_travel_time_minutes(km_to_finish)
            est_finish_ts = now + timedelta(minutes=minutes_to_finish)
            rider["km_to_finish"] = km_to_finish
            rider["est_finish_ts"] = est_finish_ts.isoformat()
        
        riders.append(rider)
    
    return {"riders": riders, "stations": stations}

def compute_critical_soc_time(rider, now):
    soc_pct = rider["soc_pct"]
    soc_remaining_before_critical = soc_pct - min_soc_thres
    
    if rider["status"] == "on_gig":
        km_to_finish = rider["km_to_finish"]
        soc_used_for_gig = km_to_finish * soc_consumption_rate
        soc_after_gig = soc_pct - soc_used_for_gig
        
        if soc_after_gig <= min_soc_thres:
            return datetime.fromisoformat(rider["est_finish_ts"].replace('Z', '+00:00')), soc_after_gig
        
        remaining_km = (soc_after_gig - min_soc_thres) / soc_consumption_rate
        minutes_to_critical = estimate_travel_time_minutes(km_to_finish) + estimate_travel_time_minutes(remaining_km)
        critical_time = now + timedelta(minutes=minutes_to_critical)
        
        return critical_time, soc_after_gig
    else:
        remaining_km = soc_remaining_before_critical / soc_consumption_rate
        minutes_to_critical = estimate_travel_time_minutes(remaining_km)
        critical_time = now + timedelta(minutes=minutes_to_critical)
        
        return critical_time, soc_pct

def find_optimal_station(rider, stations, station_schedules, now, planning_horizon_minutes=60):
    rider_lat, rider_lng = rider["lat"], rider["lng"]
    
    if rider["status"] == "on_gig":
        gig_finish_time = datetime.fromisoformat(rider["est_finish_ts"].replace('Z', '+00:00'))
        
        if (gig_finish_time - now).total_seconds() / 60 > planning_horizon_minutes:
            return None
    
    best_station = None
    min_detour = float('inf')
    best_arrival_time = None
    best_swap_start_time = None
    
    for station in stations:
        station_id = station["station_id"]
        station_lat, station_lng = station["lat"], station["lng"]
        distance_to_station = haversine_distance(rider_lat, rider_lng, station_lat, station_lng)
        
        if rider["status"] == "on_gig":
            gig_end_time = datetime.fromisoformat(rider["est_finish_ts"].replace('Z', '+00:00'))
            detour_distance = distance_to_station
            km_to_finish = rider["km_to_finish"]
            soc_used_for_gig = km_to_finish * soc_consumption_rate
            soc_after_gig = rider["soc_pct"] - soc_used_for_gig
            
            if soc_after_gig - (detour_distance * soc_consumption_rate) < min_soc_thres:
                continue
            
            travel_time_minutes = estimate_travel_time_minutes(detour_distance)
            arrival_time = gig_end_time + timedelta(minutes=travel_time_minutes)
        else:
            detour_distance = distance_to_station
            
            if rider["soc_pct"] - (detour_distance * soc_consumption_rate) < min_soc_thres:
                continue
            
            travel_time_minutes = estimate_travel_time_minutes(detour_distance)
            arrival_time = now + timedelta(minutes=travel_time_minutes)
        
        if (arrival_time - now).total_seconds() / 60 > planning_horizon_minutes:
            continue
        
        swap_start_time = find_earliest_slot(station_schedules.get(station_id, []), arrival_time)
        
        if detour_distance < min_detour:
            min_detour = detour_distance
            best_station = station
            best_arrival_time = arrival_time
            best_swap_start_time = swap_start_time
    
    if best_station:
        return {
            "rider_id": rider["rider_id"],
            "station_id": best_station["station_id"],
            "detour_km": min_detour,
            "arrival_time": best_arrival_time,
            "swap_start_time": best_swap_start_time,
            "swap_end_time": best_swap_start_time + timedelta(minutes=swap_time),
            "station_lat": best_station["lat"],
            "station_lng": best_station["lng"]
        }
    
    return None

def find_earliest_slot(station_schedule, arrival_time):
    if not station_schedule:
        return arrival_time
    
    sorted_schedule = sorted(station_schedule, key=lambda x: x["swap_start_time"])
    
    active_swaps = 0
    for swap in sorted_schedule:
        if swap["swap_start_time"] <= arrival_time < swap["swap_end_time"]:
            active_swaps += 1
    
    if active_swaps < max_station_queue:
        return arrival_time
    
    time_slots = []
    for swap in sorted_schedule:
        time_slots.append((swap["swap_start_time"], 1))
        time_slots.append((swap["swap_end_time"], -1))
    
    time_slots.sort()
    
    current_queue = 0
    earliest_available = arrival_time
    
    for time, change in time_slots:
        if time < arrival_time:
            current_queue += change
            continue
        
        current_queue += change
        if current_queue < max_station_queue:
            earliest_available = max(arrival_time, time)
            break
    
    return earliest_available

def optimize_battery_swaps(data, planning_horizon_minutes=60):
    now = datetime.now()
    riders = data["riders"]
    stations = data["stations"]
    
    station_schedules = {station["station_id"]: [] for station in stations}
    
    riders_with_critical_times = []
    for rider in riders:
        critical_time, projected_soc = compute_critical_soc_time(rider, now)
        
        if (critical_time - now).total_seconds() / 60 > planning_horizon_minutes:
            continue
        
        riders_with_critical_times.append({
            **rider,
            "critical_time": critical_time,
            "projected_soc": projected_soc
        })
    
    riders_with_critical_times.sort(key=lambda x: x["critical_time"])
    
    swap_plan = []
    
    for rider in riders_with_critical_times:
        assignment = find_optimal_station(rider, stations, station_schedules, now, planning_horizon_minutes)
        
        if assignment:
            station_id = assignment["station_id"]
            station_schedules[station_id].append({
                "rider_id": assignment["rider_id"],
                "swap_start_time": assignment["swap_start_time"],
                "swap_end_time": assignment["swap_end_time"]
            })
            
            distance_back = haversine_distance(
                assignment["station_lat"], 
                assignment["station_lng"],
                rider["lat"], 
                rider["lng"]
            )
            travel_time_back = estimate_travel_time_minutes(distance_back)
            eta_back_time = assignment["swap_end_time"] + timedelta(minutes=travel_time_back)
            
            depart_time = assignment["arrival_time"] - timedelta(minutes=estimate_travel_time_minutes(assignment["detour_km"]))
            
            swap_plan.append({
                "rider_id": rider["rider_id"],
                "station_id": assignment["station_id"],
                "depart_ts": depart_time.isoformat(),
                "arrive_ts": assignment["arrival_time"].isoformat(),
                "swap_start_ts": assignment["swap_start_time"].isoformat(),
                "swap_end_ts": assignment["swap_end_time"].isoformat(),
                "eta_back_lat": rider["lat"],
                "eta_back_lng": rider["lng"],
                "eta_back_ts": eta_back_time.isoformat(),
                "detour_km": round(assignment["detour_km"], 2)
            })
    
    return swap_plan

def save_plan_to_json(plan, filename="plan_output.json"):
    with open(filename, "w") as f:
        json.dump(plan, f, indent=2)
    return filename

def visualize_results(data, plan):
    riders = data["riders"]
    stations = data["stations"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    rider_lats = [r["lat"] for r in riders]
    rider_lngs = [r["lng"] for r in riders]
    
    idle_riders = [(r["lat"], r["lng"]) for r in riders if r["status"] == "idle"]
    busy_riders = [(r["lat"], r["lng"]) for r in riders if r["status"] == "on_gig"]
    
    station_lats = [s["lat"] for s in stations]
    station_lngs = [s["lng"] for s in stations]
    
    planned_rider_ids = [swap["rider_id"] for swap in plan]
    planned_riders = [(r["lat"], r["lng"]) for r in riders if r["rider_id"] in planned_rider_ids]
    
    ax.scatter([lng for _, lng in idle_riders], [lat for lat, _ in idle_riders], color='blue', alpha=0.5, label='Idle Riders')
    ax.scatter([lng for _, lng in busy_riders], [lat for lat, _ in busy_riders], color='green', alpha=0.5, label='Busy Riders')
    ax.scatter(station_lngs, station_lats, color='red', marker='^', s=150, label='Swap Stations')
    ax.scatter([lng for _, lng in planned_riders], [lat for lat, _ in planned_riders], color='purple', s=50, label='Riders with Planned Swaps')
    
    for station in stations:
        ax.annotate(station["station_id"], (station["lng"], station["lat"]), fontsize=12)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Battery Swap Routing - Pune')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('routing_map.png')
    plt.close()
    
    create_station_gantt(stations, plan)
    
    return "Visualizations created: routing_map.png and station_gantt.png"

def create_station_gantt(stations, plan):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    station_plans = {}
    for swap in plan:
        station_id = swap["station_id"]
        if station_id not in station_plans:
            station_plans[station_id] = []
        station_plans[station_id].append(swap)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    
    for i, station_id in enumerate(sorted(station_plans.keys())):
        swaps = station_plans[station_id]
        
        swaps.sort(key=lambda x: x["swap_start_ts"])
        
        for j, swap in enumerate(swaps):
            start_time = datetime.fromisoformat(swap["swap_start_ts"].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(swap["swap_end_ts"].replace('Z', '+00:00'))
            
            now = datetime.now()
            start_min = (start_time - now).total_seconds() / 60
            duration_min = (end_time - start_time).total_seconds() / 60
            
            ax.barh(i, duration_min, left=start_min, color=colors[i % len(colors)], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.text(start_min + duration_min/2, i, swap["rider_id"], ha='center', va='center', color='black', fontsize=8)
    
    ax.set_yticks(range(len(station_plans)))
    ax.set_yticklabels(sorted(station_plans.keys()))
    ax.set_xlabel('Minutes from Now')
    ax.set_title('Battery Swap Schedule by Station')
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('station_gantt.png')
    plt.close()
    
    return "Created station_gantt.png"

if __name__ == "__main__":
    start_time = time.time()
    
    mock_data = generate_mock_data(num_riders=100, num_stations=3)
    
    swap_plan = optimize_battery_swaps(mock_data)
    
    output_file = save_plan_to_json(swap_plan)
    print(f"Plan saved to {output_file}")
    
    vis_result = visualize_results(mock_data, swap_plan)
    print(vis_result)
    
    total_detour_km = sum(swap["detour_km"] for swap in swap_plan)
    print(f"\nSummary Statistics:")
    print(f"Total riders: {len(mock_data['riders'])}")
    print(f"Riders with planned swaps: {len(swap_plan)}")
    print(f"Total detour kilometers: {total_detour_km:.2f} km")
    print(f"Average detour per rider: {total_detour_km/len(swap_plan):.2f} km (if swaps planned)")
    
    end_time = time.time()
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")