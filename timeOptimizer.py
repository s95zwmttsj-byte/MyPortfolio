import googlemaps
import requests
from datetime import datetime, timedelta
import math


key = "key"
start = "address"
end = "address"

startWindow = "2026-08-03 04:15 PM"
endWindow = "2026-08-03 04:20 PM"

stepMins = .5

gmaps = googlemaps.Client(key= key)


def get_traffic_candidates(start, end, startDT, endDT, stepMins):
    candidates = []
    currentDT = startDT
    
    while currentDT <= endDT:
        directions = gmaps.directions(start, end, mode="driving", departure_time=currentDT)
        
        if directions:
            leg = directions[0]['legs'][0]
            trafficDuration = leg.get('duration_in_traffic', leg['duration'])['value']
            
            candidates.append({
                'departureTime': currentDT,
                'durationInTraffic': trafficDuration,
                'distance': leg['distance']['value'],
                'bounds': directions[0]['bounds']
            })
            
        currentDT += timedelta(minutes=stepMins)
        
    return candidates



def optimize_departure(start, end, startWdw, endWdw, stepMins):
    timeFormat = "%Y-%m-%d %I:%M %p"
    startDT = datetime.strptime(startWdw, timeFormat)
    endDT = datetime.strptime(endWdw, timeFormat)
    
    if endDT <= startDT:
        raise ValueError("ending window must be after starting window.")

    durationMins = (endDT - startDT).total_seconds() / 60.0
    minAllowedStep = durationMins / 10.0
    
    if stepMins < minAllowedStep:
        print(f"Step size adjusted from {stepMins} min to minimum 1/10th window duration ({minAllowedStep:.1f} min).")
        stepMins = minAllowedStep

    print(f"\nEvaluating route from {startDT.strftime('%I:%M %p')} to {endDT.strftime('%I:%M %p')} (Interval: {stepMins:.1f} mins)...")
    
    candidates = get_traffic_candidates(start, end, startDT, endDT, stepMins)
    
    if not candidates:
        print("No route candidates found.")
        return
    

    minDuration = min(c['durationInTraffic'] for c in candidates)
    best = None
    lowestScore = float('inf')

    print("\nResults:")
    for c in candidates:
        trafficDelay = c['durationInTraffic'] - minDuration
        
        w1 = 1.0
        w2 = 120.0
        
        trafficPenalty = (trafficDelay / 60.0) * w2
        score = (trafficDelay * w1) + trafficPenalty
        
        departureString = c['departureTime'].strftime('%I:%M %p')
        travelMins = round(c['durationInTraffic'] / 60, 1)
        print(f"Leave at {departureString}, Travel Time: {travelMins} min, Stopping Risk: {round(score, 1)}")
        
        if score < lowestScore:
            lowestScore = score
            best = c

    bestTime = best['departureTime'].strftime('%I:%M %p')
    bestDur = round(best['durationInTraffic'] / 60, 1)
    
    print(f" Best time to leave: {bestTime}")
    print(f" Projected travel time: {bestDur} mins")

if __name__ == "__main__":
    optimize_departure(start, end, startWindow, endWindow, stepMins)
