import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_medical_events(num_sequences, events_per_sequence, start_date, event_types):
    all_sequences = []
    for seq_id in range(num_sequences):
        sequence = []
        current_time = start_date
        for _ in range(events_per_sequence):
            event_type = random.choice(event_types)
            event_time = current_time + timedelta(hours=random.randint(1, 24))
            sequence.append([seq_id, event_time, event_type])
            current_time = event_time
        all_sequences.extend(sequence)
    return all_sequences

# Parameters
num_sequences = 5
events_per_sequence = 50
start_date = datetime(2023, 1, 1)
event_types = ['medication', 'doctor_visit', 'lab_test', 'appointment']

# Generate and save the synthetic events
synthetic_events = generate_medical_events(num_sequences, events_per_sequence, start_date, event_types)
df = pd.DataFrame(synthetic_events, columns=['sequence_id', 'event_time', 'event_type'])
df.to_csv('data/medical_events.csv', index=False)
