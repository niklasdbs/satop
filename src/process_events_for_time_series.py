import pandas as pd
import numpy as np
from tqdm import tqdm
from envs.enums import ParkingStatus, EventType
import itertools
events = pd.read_csv("../data/2019/downtown_event.csv")

resources = pd.read_csv("../data/2019/downtown_resources.csv")

n_resources = len(resources)


all_data = {}
all_lengths = {}

cyclical_time_encoding = True
n_features = 18



def encode_resource_event(day, time, hour, max_parking_duration_seconds, event_type, time_since_last_event, month, day_of_week):
    encoded_data = np.zeros((1 + n_features), dtype=np.float32)

            
    offset = 0
    encoded_data[offset] = time #set the time (which is always at position 0)
    offset +=1
            
    encoded_data[offset + event_type] = 1 #encode type of event
    offset += 4
    
    encoded_data[offset] = 0 #indicate regular event
    offset += 1

            
    encoded_data[offset] = max_parking_duration_seconds / 3600.0
    offset += 1
            
    encoded_data[offset] = time / 3600.0
    offset += 1

    encoded_data[offset] = time_since_last_event / 3600.0
    offset += 1


    if cyclical_time_encoding:
                #encode hour
        encoded_data[offset] = np.sin(2 * np.pi * hour/24.0)
        offset += 1

        encoded_data[offset] = np.cos(2 * np.pi * hour/24.0)
        offset += 1
            
        #encode day
        encoded_data[offset] = np.sin(2 * np.pi * (day-1)/365.0)
        offset += 1

        encoded_data[offset] = np.cos(2 * np.pi * (day-1)/365.0)
        offset += 1

        encoded_data[offset] = np.sin(2 * np.pi * (month-1)/12.0)
        offset += 1

        encoded_data[offset] = np.cos(2 * np.pi * (month-1)/12.0)
        offset += 1
                
                #encode day of week
        encoded_data[offset] = np.sin(2 * np.pi * day_of_week/7.0)
        offset += 1

        encoded_data[offset] = np.cos(2 * np.pi * day_of_week/7.0)
        offset += 1
                
                
                #encode seconds_since_midnight
        encoded_data[offset] = np.sin(2 * np.pi * time/(24*60*60))
        offset += 1

        encoded_data[offset] = np.cos(2 * np.pi * time/(24*60*60))
        offset += 1
    return encoded_data

for day, events_day in tqdm(events.groupby("DayOfYear")):
    
    if len(events_day) == 1 or len(events_day)==0:
        raise RuntimeError("oh no")
        
    events_by_r = events_day.groupby("ResourceID")
    
    per_day_data = []
    
    event_for_start_vars = events_day.iloc[0]
    month = event_for_start_vars["Month"]
    day_of_week = event_for_start_vars["DayOfWeek"]

    for _, resource in resources.iterrows():
        data = []

        r_id = int(resource["resource_id"])


        
        time = 0
        hour = 0
        arrival_time = 0
        max_parking_duration_seconds = 0
        current_state = ParkingStatus.FREE
        time_since_last_event = 0
        
        #start event
        event_type = 3 #indicate starte event
        encoded_data = encode_resource_event(day, time, hour, max_parking_duration_seconds, event_type, time_since_last_event, month, day_of_week)

        data.append(encoded_data)
        
        
        i = 1

        if not r_id in events_by_r.groups:
            per_day_data.append(np.stack(data))

            continue
        
        rel_events = events_by_r.get_group(r_id)
        
        rel_events = rel_events.sort_values("TimeOfDay")

        for _, event in rel_events.iterrows():

            event_type = event["EventType"]
            
            if event_type == EventType.VIOLATION and (current_state == ParkingStatus.IN_VIOLATION or current_state == ParkingStatus.FREE):
                continue
            

            time_since_last_event = time - event["TimeOfDay"]
            
            time = event["TimeOfDay"]
            hour = event["Hour"]
            month = event["Month"]
            day_of_week = event["DayOfWeek"]
            
            if event_type == EventType.ARRIVAL:
                current_state = ParkingStatus.OCCUPIED
                arrival_time = time
                max_parking_duration_seconds = event["MaxSeconds"]
            elif event_type == EventType.DEPARTURE:
                arrival_time = 0
                current_state = ParkingStatus.FREE
                max_parking_duration_seconds = event["MaxSeconds"]
            elif event_type == EventType.VIOLATION:
                current_state = ParkingStatus.IN_VIOLATION
                time_last_violation = time

            
            encoded_data = encode_resource_event(day, time, hour, max_parking_duration_seconds, event_type, time_since_last_event, month, day_of_week)



            
            
            i += 1
            
            data.append(encoded_data)

        
        per_day_data.append(np.stack(data))
        
    lengths = [x.shape[0] for x in per_day_data]
    max_len = max(lengths) #TODO add safety bounds
    pad_values = np.zeros((1 + n_features), dtype=np.float32)
    pad_values[0] = np.finfo(np.float32).max #set padded value times to max value
    
    #final_np = np.stack([np.pad(x, [ (0,max_len-x.shape[0]), (0, 0)], "constant", constant_values=0) for x in per_day_data ])

    final_np = np.stack([np.concatenate([x, pad_values[None, :].repeat(max_len-x.shape[0], 0)], axis=0) for x in per_day_data])
    
    
    all_data[str(day)] = final_np
    all_lengths[str(day)] = lengths

np.savez_compressed("../data/2019/downtown_time_series.npz", **all_data)
np.savez_compressed("../data/2019/downtown_lengths.npz", **all_lengths)

#np.where(np.expand_dims((final_np[:,:,0]<10*60*60),-1), final_np, 0).shape