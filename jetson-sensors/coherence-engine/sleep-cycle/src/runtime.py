import time
import random
import pandas as pd
from pathlib import Path

LOG_PATH = Path('event_log.parquet')

def generate_event():
    return {
        'timestamp': time.time(),
        'vision_latent': [random.random() for _ in range(8)],
        'imu_latent': [random.random() for _ in range(4)],
        'coherence_score': random.random()
    }

def main():
    print("Simulating coherence engine event stream...")
    events = []
    for _ in range(50):  # Simulate 50 events
        event = generate_event()
        events.append(event)
        time.sleep(0.1)
    df = pd.DataFrame(events)
    df.to_parquet(LOG_PATH, engine='pyarrow')
    print(f"Saved {len(events)} events to {LOG_PATH}")

if __name__ == '__main__':
    main()
