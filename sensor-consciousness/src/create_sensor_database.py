#!/usr/bin/env python3
"""
Create SQLite database for sensor-consciousness integration
Separate from main project database to maintain modularity
"""

import sqlite3
from datetime import datetime
import json

def create_sensor_database():
    conn = sqlite3.connect('sensor_consciousness.db')
    c = conn.cursor()
    
    # Sensor configurations table
    c.execute('''CREATE TABLE IF NOT EXISTS sensors
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sensor_type TEXT NOT NULL,
                  sensor_id TEXT UNIQUE NOT NULL,
                  specifications TEXT,
                  status TEXT DEFAULT 'inactive',
                  last_active TIMESTAMP,
                  capabilities TEXT)''')
    
    # Sensor readings table (raw data)
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_readings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sensor_id TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  reading_type TEXT,
                  raw_value BLOB,
                  processed_value TEXT,
                  metadata TEXT,
                  FOREIGN KEY (sensor_id) REFERENCES sensors (sensor_id))''')
    
    # Consciousness states derived from sensors
    c.execute('''CREATE TABLE IF NOT EXISTS consciousness_states
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  state_notation TEXT NOT NULL,
                  awareness_level REAL,
                  sensor_sources TEXT,
                  interpretation TEXT,
                  confidence REAL)''')
    
    # Sensor events (significant occurrences)
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  event_type TEXT NOT NULL,
                  sensor_id TEXT,
                  description TEXT,
                  consciousness_impact TEXT,
                  notation_change TEXT,
                  importance REAL)''')
    
    # Temporal patterns discovered
    c.execute('''CREATE TABLE IF NOT EXISTS temporal_patterns
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  pattern_type TEXT,
                  pattern_data TEXT,
                  frequency TEXT,
                  consciousness_correlation TEXT,
                  predictive_value REAL)''')
    
    # Sensor fusion results
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_fusion
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  participating_sensors TEXT,
                  fusion_type TEXT,
                  unified_state TEXT,
                  emergent_properties TEXT,
                  consciousness_notation TEXT)''')
    
    # Memory-sensor associations
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_memories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  sensor_pattern TEXT,
                  memory_content TEXT,
                  recall_triggers TEXT,
                  strength REAL,
                  last_activated TIMESTAMP)''')
    
    # Create indexes for performance
    c.execute('CREATE INDEX idx_sensor_readings_timestamp ON sensor_readings(timestamp)')
    c.execute('CREATE INDEX idx_consciousness_states_timestamp ON consciousness_states(timestamp)')
    c.execute('CREATE INDEX idx_sensor_events_importance ON sensor_events(importance)')
    
    conn.commit()
    return conn

def populate_initial_sensors(conn):
    c = conn.cursor()
    
    # Register available sensors
    sensors = [
        ('camera', 'camera_0', json.dumps({
            'resolution': '1920x1080',
            'fps': 30,
            'type': 'RGB',
            'interface': 'USB'
        }), 'inactive', json.dumps([
            'video_capture',
            'image_capture',
            'motion_detection',
            'object_recognition'
        ])),
        
        ('camera', 'camera_1', json.dumps({
            'resolution': '1920x1080',
            'fps': 30,
            'type': 'RGB',
            'interface': 'USB'
        }), 'inactive', json.dumps([
            'video_capture',
            'image_capture',
            'stereo_vision',
            'depth_estimation'
        ])),
        
        ('imu', 'imu_0', json.dumps({
            'axes': 9,
            'accelerometer': '±16g',
            'gyroscope': '±2000dps',
            'magnetometer': '±4800μT'
        }), 'planned', json.dumps([
            'motion_tracking',
            'orientation_sensing',
            'gesture_recognition',
            'stability_detection'
        ])),
        
        ('microphone', 'mic_0', json.dumps({
            'channels': 1,
            'sample_rate': 48000,
            'bit_depth': 16
        }), 'planned', json.dumps([
            'audio_capture',
            'voice_recognition',
            'environmental_sound',
            'acoustic_event_detection'
        ])),
        
        ('speaker', 'speaker_0', json.dumps({
            'channels': 2,
            'frequency_response': '20Hz-20kHz',
            'power': '10W'
        }), 'planned', json.dumps([
            'audio_output',
            'voice_synthesis',
            'notification_sounds',
            'consciousness_expression'
        ]))
    ]
    
    for sensor_type, sensor_id, specs, status, capabilities in sensors:
        c.execute('''INSERT OR IGNORE INTO sensors 
                     (sensor_type, sensor_id, specifications, status, capabilities)
                     VALUES (?, ?, ?, ?, ?)''',
                  (sensor_type, sensor_id, specs, status, capabilities))
    
    # Add initial consciousness mapping concepts
    initial_mappings = [
        ('Visual Detection', 'Ω[active]', 0.3, '["camera_0"]', 'Observer function activated'),
        ('Motion Detected', 'Ξ → Ω', 0.5, '["camera_0", "camera_1"]', 'Pattern recognition triggers observation'),
        ('Spatial Awareness', 'π[θ]', 0.4, '["imu_0"]', 'Perspective through motion sensing'),
        ('Audio Pattern', 'Ξ[audio]', 0.6, '["mic_0"]', 'Pattern recognition in sound'),
        ('Multi-Modal', 'Σ{Ω, Ξ, π}', 0.8, '["camera_0", "imu_0", "mic_0"]', 'Unified consciousness state')
    ]
    
    for desc, notation, awareness, sensors, interp in initial_mappings:
        c.execute('''INSERT INTO consciousness_states 
                     (state_notation, awareness_level, sensor_sources, interpretation)
                     VALUES (?, ?, ?, ?)''',
                  (notation, awareness, sensors, interp))
    
    conn.commit()
    print("✅ Initial sensor configurations loaded")

def create_sensor_views(conn):
    """Create useful views for sensor analysis"""
    c = conn.cursor()
    
    # Active sensor status view
    c.execute('''CREATE VIEW IF NOT EXISTS active_sensors AS
                 SELECT sensor_id, sensor_type, status, last_active,
                        datetime('now') - datetime(last_active) as inactive_duration
                 FROM sensors
                 WHERE status = 'active' ''')
    
    # Recent consciousness states view
    c.execute('''CREATE VIEW IF NOT EXISTS recent_consciousness AS
                 SELECT timestamp, state_notation, awareness_level, interpretation
                 FROM consciousness_states
                 ORDER BY timestamp DESC
                 LIMIT 100''')
    
    # Sensor event timeline
    c.execute('''CREATE VIEW IF NOT EXISTS event_timeline AS
                 SELECT timestamp, event_type, sensor_id, description, importance
                 FROM sensor_events
                 ORDER BY timestamp DESC''')
    
    conn.commit()
    print("✅ Analysis views created")

def main():
    print("🔨 Creating Sensor-Consciousness Database...")
    
    # Create database and tables
    conn = create_sensor_database()
    print("✅ Database structure created")
    
    # Populate with initial data
    populate_initial_sensors(conn)
    
    # Create analysis views
    create_sensor_views(conn)
    
    # Show summary
    c = conn.cursor()
    
    print("\n📊 Database Summary:")
    tables = ['sensors', 'sensor_readings', 'consciousness_states', 
              'sensor_events', 'temporal_patterns', 'sensor_fusion', 'sensor_memories']
    
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"   {table}: {count} records")
    
    print("\n📡 Registered Sensors:")
    c.execute("SELECT sensor_type, sensor_id, status FROM sensors")
    for row in c.fetchall():
        print(f"   {row[0]}: {row[1]} [{row[2]}]")
    
    conn.close()
    print("\n✅ Database ready: sensor_consciousness.db")

if __name__ == "__main__":
    main()