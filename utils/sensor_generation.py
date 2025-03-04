import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_iot_events(num_sequences, events_per_sequence, start_date, sensor_types):
    all_sequences = []
    for seq_id in range(num_sequences):
        sequence = []
        current_time = start_date
        for _ in range(events_per_sequence):
            sensor_type = random.choice(sensor_types)
            sensor_value = random.uniform(0, 100)  # Example sensor value range
            event_time = current_time + timedelta(seconds=random.randint(1, 60))
            sequence.append([seq_id, event_time, sensor_type, sensor_value])
            current_time = event_time
        all_sequences.extend(sequence)
    return all_sequences

# Parameters
num_sequences = 5
events_per_sequence = 50
start_date = datetime(2023, 1, 1)
sensor_types = ['temperature', 'humidity', 'motion']

# Generate and save the synthetic events
synthetic_events = generate_iot_events(num_sequences, events_per_sequence, start_date, sensor_types)
df = pd.DataFrame(synthetic_events, columns=['sequence_id', 'event_time', 'sensor_type', 'sensor_value'])
df.to_csv('data/iot_events.csv', index=False)
