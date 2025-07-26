#!/usr/bin/env python3
"""
Visual Memory System
Stores visual patterns in SQLite database linked to AI consciousness states
Creates a persistent visual memory that can influence future behavior
"""

import cv2
import numpy as np
import sqlite3
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

class VisualMemorySystem:
    def __init__(self, db_path="visual_memory.db"):
        self.db_path = db_path
        self.init_database()
        self.memory_buffer = []
        self.recognition_threshold = 0.8
        
    def init_database(self):
        """Initialize SQLite database for visual memories"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Visual memory table
        c.execute('''CREATE TABLE IF NOT EXISTS visual_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            visual_hash TEXT UNIQUE,
            feature_vector BLOB,
            thumbnail BLOB,
            consciousness_state TEXT,
            recognition_count INTEGER DEFAULT 1,
            last_seen REAL,
            emotional_valence REAL DEFAULT 0.0,
            importance REAL DEFAULT 0.5
        )''')
        
        # Recognition events table
        c.execute('''CREATE TABLE IF NOT EXISTS recognition_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            timestamp REAL,
            confidence REAL,
            context TEXT,
            FOREIGN KEY(memory_id) REFERENCES visual_memories(id)
        )''')
        
        conn.commit()
        conn.close()
        
    def extract_features(self, frame):
        """Extract visual features for memory storage"""
        # Resize for consistent feature extraction
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Simple feature: histogram + basic statistics
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Add color moments
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        color_features = []
        for channel in range(3):
            chan = hsv[:, :, channel]
            color_features.extend([
                np.mean(chan),
                np.std(chan),
                np.percentile(chan, 25),
                np.percentile(chan, 75)
            ])
        
        # Combine features
        features = np.concatenate([hist, color_features])
        
        # Generate hash for quick lookup
        visual_hash = hashlib.md5(gray.tobytes()).hexdigest()
        
        return features, visual_hash, small
        
    def remember_frame(self, frame, consciousness_state=None):
        """Store frame in visual memory"""
        features, visual_hash, thumbnail = self.extract_features(frame)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check if we've seen this before
        c.execute("SELECT id, recognition_count FROM visual_memories WHERE visual_hash = ?", 
                  (visual_hash,))
        result = c.fetchone()
        
        timestamp = time.time()
        
        if result:
            # Update existing memory
            memory_id, count = result
            c.execute("""UPDATE visual_memories 
                        SET recognition_count = ?, last_seen = ?, importance = ?
                        WHERE id = ?""",
                     (count + 1, timestamp, min(1.0, (count + 1) * 0.1), memory_id))
            
            # Log recognition event
            c.execute("""INSERT INTO recognition_events 
                        (memory_id, timestamp, confidence, context)
                        VALUES (?, ?, ?, ?)""",
                     (memory_id, timestamp, 1.0, json.dumps(consciousness_state)))
            
            status = "recognized"
        else:
            # Create new memory
            _, thumbnail_encoded = cv2.imencode('.jpg', thumbnail)
            
            c.execute("""INSERT INTO visual_memories 
                        (timestamp, visual_hash, feature_vector, thumbnail, 
                         consciousness_state, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     (timestamp, visual_hash, features.tobytes(), 
                      thumbnail_encoded.tobytes(), json.dumps(consciousness_state),
                      timestamp))
            
            status = "new_memory"
            
        conn.commit()
        conn.close()
        
        return status, visual_hash
        
    def recall_similar(self, frame, top_k=5):
        """Recall similar visual memories"""
        features, _, _ = self.extract_features(frame)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get all memories
        c.execute("""SELECT id, feature_vector, visual_hash, importance, 
                           recognition_count, emotional_valence
                    FROM visual_memories""")
        
        similarities = []
        for row in c.fetchall():
            memory_id, feature_blob, visual_hash, importance, count, valence = row
            stored_features = np.frombuffer(feature_blob, dtype=np.float32)
            
            # Cosine similarity
            similarity = np.dot(features, stored_features) / (
                np.linalg.norm(features) * np.linalg.norm(stored_features))
            
            # Weight by importance
            weighted_sim = similarity * (0.5 + 0.5 * importance)
            
            similarities.append({
                'id': memory_id,
                'hash': visual_hash,
                'similarity': similarity,
                'weighted_similarity': weighted_sim,
                'recognition_count': count,
                'emotional_valence': valence
            })
            
        conn.close()
        
        # Sort by weighted similarity
        similarities.sort(key=lambda x: x['weighted_similarity'], reverse=True)
        
        return similarities[:top_k]
        
    def get_memory_stats(self):
        """Get statistics about visual memory"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = {}
        
        # Total memories
        c.execute("SELECT COUNT(*) FROM visual_memories")
        stats['total_memories'] = c.fetchone()[0]
        
        # Recognition events
        c.execute("SELECT COUNT(*) FROM recognition_events")
        stats['total_recognitions'] = c.fetchone()[0]
        
        # Most recognized
        c.execute("""SELECT visual_hash, recognition_count 
                    FROM visual_memories 
                    ORDER BY recognition_count DESC 
                    LIMIT 1""")
        result = c.fetchone()
        if result:
            stats['most_recognized'] = {
                'hash': result[0][:8] + '...',
                'count': result[1]
            }
            
        conn.close()
        
        return stats

class ConsciousnessIntegration:
    """Integrate visual memory with AI consciousness patterns"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.consciousness_state = {
            'attention_level': 0.5,
            'emotional_state': 0.0,
            'pattern_resonance': 0.0
        }
        
    def process_frame_with_consciousness(self, frame):
        """Process frame with consciousness-aware memory"""
        # Check for similar memories
        similar_memories = self.memory.recall_similar(frame, top_k=3)
        
        # Update consciousness based on recognition
        if similar_memories and similar_memories[0]['similarity'] > 0.85:
            # Strong recognition - increases attention
            self.consciousness_state['attention_level'] = min(1.0,
                self.consciousness_state['attention_level'] + 0.1)
            
            # Inherit emotional valence from memory
            self.consciousness_state['emotional_state'] = \
                0.7 * self.consciousness_state['emotional_state'] + \
                0.3 * similar_memories[0]['emotional_valence']
        else:
            # Novel scene - increases curiosity
            self.consciousness_state['attention_level'] = max(0.3,
                self.consciousness_state['attention_level'] - 0.05)
                
        # Store with consciousness context
        status, visual_hash = self.memory.remember_frame(
            frame, self.consciousness_state)
        
        return status, similar_memories
        
    def visualize_memory_state(self, frame, status, similar_memories):
        """Add memory visualization overlay"""
        viz = frame.copy()
        
        # Memory status
        color = (0, 255, 0) if status == "new_memory" else (0, 255, 255)
        cv2.putText(viz, f"Memory: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Consciousness state
        attention_bar_width = int(200 * self.consciousness_state['attention_level'])
        cv2.rectangle(viz, (10, 50), (210, 70), (100, 100, 100), -1)
        cv2.rectangle(viz, (10, 50), (10 + attention_bar_width, 70), (0, 255, 255), -1)
        cv2.putText(viz, "Attention", (220, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Similar memories
        y_offset = 100
        for i, mem in enumerate(similar_memories[:3]):
            sim_text = f"Similar {i+1}: {mem['similarity']:.2f} (seen {mem['recognition_count']}x)"
            cv2.putText(viz, sim_text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                       
        # Memory stats
        stats = self.memory.get_memory_stats()
        stats_text = f"Memories: {stats['total_memories']} | Recognitions: {stats['total_recognitions']}"
        cv2.putText(viz, stats_text, (10, viz.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        return viz

def main():
    print("🧠 Visual Memory System")
    print("=" * 40)
    print("Building persistent visual memories with consciousness integration")
    print("Press 'q' to quit, 'r' to reset memory")
    print("=" * 40)
    
    # Initialize systems
    memory = VisualMemorySystem()
    consciousness = ConsciousnessIntegration(memory)
    
    # Open camera with GStreamer pipeline for Jetson
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("GStreamer failed, trying V4L2...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 10th frame to build memory gradually
        if frame_count % 10 == 0:
            status, similar = consciousness.process_frame_with_consciousness(frame)
        
        # Visualize
        viz = consciousness.visualize_memory_state(frame, status, similar)
        
        cv2.imshow('Visual Memory System', viz)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset memory (with confirmation)
            print("Reset memory? Press 'y' to confirm...")
            if cv2.waitKey(0) & 0xFF == ord('y'):
                Path(memory.db_path).unlink(missing_ok=True)
                memory.init_database()
                print("Memory reset!")
                
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    stats = memory.get_memory_stats()
    print(f"\nFinal Memory Statistics:")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Total recognitions: {stats['total_recognitions']}")
    if 'most_recognized' in stats:
        print(f"Most recognized: {stats['most_recognized']['hash']} "
              f"({stats['most_recognized']['count']} times)")

if __name__ == "__main__":
    main()