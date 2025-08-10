#!/usr/bin/env python3
"""
Coherence Engine Dashboard - Real-time visualization
"""

import sys
import os
import time
import threading
import numpy as np
import cv2
from datetime import datetime
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensors'))

from coherence_engine import CoherenceEngine
from sensors.memory_sensor import MemorySensor

class CoherenceDashboard:
    """Real-time dashboard for coherence engine"""
    
    def __init__(self, width=1600, height=900):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.running = False
        
        # Engine reference
        self.engine = None
        self.engine_thread = None
        
        # History tracking for graphs
        self.confidence_history = deque(maxlen=200)
        self.attention_history = deque(maxlen=200)
        self.context_history = deque(maxlen=200)
        self.sensor_trust_history = {}  # sensor_name -> deque
        
        # Colors
        self.colors = {
            'bg': (20, 20, 20),
            'panel': (40, 40, 40),
            'text': (255, 255, 255),
            'stable': (100, 255, 100),
            'moving': (255, 255, 100),
            'unstable': (255, 100, 100),
            'novel': (100, 100, 255),
            'high': (255, 50, 50),
            'medium': (255, 200, 50),
            'low': (100, 255, 100),
            'graph_bg': (30, 30, 30),
            'grid': (50, 50, 50)
        }
        
        # Context to number mapping for graphing
        self.context_to_num = {
            'stable': 0,
            'moving': 1,
            'unstable': 2,
            'novel': 3
        }
        
    def init_engine(self):
        """Initialize coherence engine with sensors"""
        memory_path = os.path.join(os.path.dirname(__file__), "memory")
        self.engine = CoherenceEngine(memory_path=memory_path)
        
        # Register memory sensor
        memory_sensor = MemorySensor(memory_path=memory_path)
        self.engine.register_sensor(memory_sensor)
        
        # Initialize sensor trust history
        for name in self.engine.sensors.keys():
            self.sensor_trust_history[name] = deque(maxlen=200)
    
    def run_engine(self):
        """Run engine in background thread"""
        while self.running:
            # Step engine
            reality_field = self.engine.step()
            
            # Update history
            self.confidence_history.append(reality_field.overall_confidence)
            self.attention_history.append(self.engine.current_context.attention_level)
            self.context_history.append(
                self.context_to_num.get(self.engine.current_context.state, 0)
            )
            
            # Update sensor trust
            for name, sensor in self.engine.sensors.items():
                if name not in self.sensor_trust_history:
                    self.sensor_trust_history[name] = deque(maxlen=200)
                self.sensor_trust_history[name].append(sensor.trust_score)
            
            time.sleep(0.1)  # 10Hz update
    
    def draw_header(self):
        """Draw dashboard header"""
        cv2.rectangle(self.canvas, (0, 0), (self.width, 60), self.colors['panel'], -1)
        
        # Title
        cv2.putText(self.canvas, "COHERENCE ENGINE DASHBOARD", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['text'], 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(self.canvas, timestamp, (self.width - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 1)
    
    def draw_context_panel(self, x, y, w, h):
        """Draw current context state panel"""
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.putText(self.canvas, "CONTEXT STATE", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
        
        if self.engine:
            context = self.engine.current_context
            
            # State indicator (large)
            state_color = self.colors.get(context.state, self.colors['stable'])
            cv2.rectangle(self.canvas, (x+20, y+50), (x+w-20, y+120), state_color, -1)
            cv2.putText(self.canvas, context.state.upper(), (x+w//2-60, y+95),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Metrics
            cv2.putText(self.canvas, f"Attention: {context.attention_level:.2f}", 
                       (x+20, y+150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            cv2.putText(self.canvas, f"Confidence: {context.confidence:.2f}", 
                       (x+20, y+180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            
            # Active sensors
            cv2.putText(self.canvas, "Active Sensors:", (x+20, y+210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            for i, sensor_name in enumerate(context.active_sensors[:5]):
                cv2.putText(self.canvas, f"  • {sensor_name}", (x+20, y+235+i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['low'], 1)
    
    def draw_sensor_panel(self, x, y, w, h):
        """Draw sensor status panel"""
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.putText(self.canvas, "SENSOR STATUS", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
        
        if self.engine:
            y_offset = 60
            for name, sensor in self.engine.sensors.items():
                # Sensor name
                cv2.putText(self.canvas, name, (x+20, y+y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
                
                # Trust bar
                trust_width = int(sensor.trust_score * 150)
                bar_color = (0, int(255 * sensor.trust_score), int(255 * (1-sensor.trust_score)))
                cv2.rectangle(self.canvas, (x+150, y+y_offset-15), 
                             (x+150+trust_width, y+y_offset), bar_color, -1)
                cv2.rectangle(self.canvas, (x+150, y+y_offset-15), 
                             (x+300, y+y_offset), self.colors['grid'], 1)
                
                # Trust value
                cv2.putText(self.canvas, f"{sensor.trust_score:.2f}", 
                           (x+310, y+y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                
                # Status indicator
                status_color = self.colors['low'] if sensor.is_active else self.colors['high']
                cv2.circle(self.canvas, (x+370, y+y_offset-7), 5, status_color, -1)
                
                y_offset += 35
    
    def draw_triggers_panel(self, x, y, w, h):
        """Draw attention triggers panel"""
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.putText(self.canvas, "ATTENTION TRIGGERS", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
        
        if self.engine and self.engine.reality_history:
            latest_field = self.engine.reality_history[-1]
            triggers = latest_field.attention_triggers
            
            if triggers:
                y_offset = 60
                for trigger in triggers[:8]:  # Show max 8 triggers
                    # Trigger type
                    cv2.putText(self.canvas, trigger['type'], (x+20, y+y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                    
                    # Severity indicator
                    severity_color = self.colors.get(trigger['severity'], self.colors['medium'])
                    cv2.rectangle(self.canvas, (x+200, y+y_offset-15), 
                                 (x+w-20, y+y_offset), severity_color, -1)
                    cv2.putText(self.canvas, trigger['severity'].upper(), 
                               (x+210, y+y_offset-2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    y_offset += 25
            else:
                cv2.putText(self.canvas, "No triggers", (x+20, y+60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['low'], 1)
    
    def draw_graph(self, x, y, w, h, data, title, color=(100, 255, 100), y_range=(0, 1)):
        """Draw a time series graph"""
        # Background
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['graph_bg'], -1)
        
        # Title
        cv2.putText(self.canvas, title, (x+10, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Grid lines
        for i in range(1, 4):
            y_grid = y + int(h * i / 4)
            cv2.line(self.canvas, (x, y_grid), (x+w, y_grid), self.colors['grid'], 1)
        
        # Plot data
        if len(data) > 1:
            # Scale data to graph dimensions
            points = []
            for i, value in enumerate(data):
                # Normalize value to y_range
                norm_value = (value - y_range[0]) / (y_range[1] - y_range[0])
                norm_value = max(0, min(1, norm_value))  # Clamp to 0-1
                
                px = x + int(i * w / len(data))
                py = y + h - int(norm_value * (h - 30))
                points.append((px, py))
            
            # Draw line
            for i in range(1, len(points)):
                cv2.line(self.canvas, points[i-1], points[i], color, 2)
            
            # Current value
            if data:
                current_val = data[-1]
                cv2.putText(self.canvas, f"{current_val:.2f}", 
                           (x+w-60, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_memory_panel(self, x, y, w, h):
        """Draw memory sensor specific panel"""
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.putText(self.canvas, "MEMORY SENSOR", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
        
        if self.engine and 'memory_sensor' in self.engine.sensors:
            memory = self.engine.sensors['memory_sensor']
            if memory.last_reading:
                data = memory.last_reading.data
                
                # Working memory size
                cv2.putText(self.canvas, f"Working Memory: {data.get('working_memory_size', 0)} items",
                           (x+20, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                
                # Pattern cache
                cv2.putText(self.canvas, f"Known Patterns: {data.get('pattern_cache_size', 0)}",
                           (x+20, y+85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                
                # Patterns detected
                patterns = data.get('patterns_detected', [])
                if patterns:
                    cv2.putText(self.canvas, "Detected Patterns:", (x+20, y+115),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                    for i, pattern in enumerate(patterns[:3]):
                        cv2.putText(self.canvas, f"  • {pattern}", (x+20, y+140+i*20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['low'], 1)
                
                # Prediction
                prediction = data.get('prediction', {})
                if prediction:
                    cv2.putText(self.canvas, f"Prediction: {prediction.get('likely_next_state', 'unknown')}",
                               (x+20, y+220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['novel'], 1)
                    cv2.putText(self.canvas, f"Confidence: {prediction.get('confidence', 0):.1%}",
                               (x+20, y+245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['novel'], 1)
    
    def update_display(self):
        """Update the entire dashboard display"""
        # Clear canvas
        self.canvas[:] = self.colors['bg']
        
        # Draw header
        self.draw_header()
        
        # Left column - Context and Sensors
        self.draw_context_panel(20, 80, 400, 350)
        self.draw_sensor_panel(20, 450, 400, 300)
        
        # Middle column - Graphs
        self.draw_graph(440, 80, 350, 150, self.confidence_history, 
                       "CONFIDENCE", self.colors['stable'])
        self.draw_graph(440, 250, 350, 150, self.attention_history, 
                       "ATTENTION", self.colors['unstable'])
        self.draw_graph(440, 420, 350, 150, self.context_history, 
                       "CONTEXT STATE", self.colors['novel'], y_range=(0, 3))
        
        # Sensor trust graphs
        y_offset = 590
        for name, history in list(self.sensor_trust_history.items())[:2]:
            self.draw_graph(440, y_offset, 350, 120, history, 
                           f"{name.upper()} TRUST", (255, 200, 100))
            y_offset += 130
        
        # Right column - Triggers and Memory
        self.draw_triggers_panel(810, 80, 350, 320)
        self.draw_memory_panel(810, 420, 350, 290)
        
        # Reality field visualization
        self.draw_reality_field(1180, 80, 400, 630)
        
        # Stats bar at bottom
        self.draw_stats_bar()
    
    def draw_reality_field(self, x, y, w, h):
        """Visualize the reality field"""
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.putText(self.canvas, "REALITY FIELD", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
        
        if self.engine and self.engine.reality_history:
            field = self.engine.reality_history[-1]
            
            # Visualize as concentric circles representing sensor contributions
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw base circle
            cv2.circle(self.canvas, (center_x, center_y), 150, self.colors['grid'], 1)
            
            # Draw sensor contributions as arcs/sections
            if field.sensor_contributions:
                total = sum(field.sensor_contributions.values())
                if total > 0:
                    start_angle = 0
                    for name, contrib in field.sensor_contributions.items():
                        # Calculate arc size based on contribution
                        arc_angle = int(360 * (contrib / total))
                        
                        # Get color based on sensor type
                        sensor_type = self.engine.sensors[name].sensor_type
                        if sensor_type == 'memory':
                            color = self.colors['novel']
                        elif sensor_type == 'vision':
                            color = self.colors['stable']
                        elif sensor_type == 'imu':
                            color = self.colors['moving']
                        else:
                            color = self.colors['text']
                        
                        # Draw arc
                        radius = int(150 * (0.3 + contrib * 2))  # Scale by contribution
                        cv2.ellipse(self.canvas, (center_x, center_y), 
                                   (radius, radius), 0, start_angle, 
                                   start_angle + arc_angle, color, -1)
                        
                        start_angle += arc_angle
            
            # Central confidence indicator
            conf_radius = int(field.overall_confidence * 50)
            conf_color = (
                int(255 * (1 - field.overall_confidence)),
                int(255 * field.overall_confidence),
                0
            )
            cv2.circle(self.canvas, (center_x, center_y), conf_radius, conf_color, -1)
            
            # Overall confidence text
            cv2.putText(self.canvas, f"{field.overall_confidence:.1%}",
                       (center_x - 30, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def draw_stats_bar(self):
        """Draw statistics bar at bottom"""
        y = self.height - 40
        cv2.rectangle(self.canvas, (0, y), (self.width, self.height), 
                     self.colors['panel'], -1)
        
        if self.engine:
            # Steps
            steps = len(self.engine.reality_history)
            cv2.putText(self.canvas, f"Steps: {steps}", (20, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # FPS (approximate)
            fps = 10  # We're running at 10Hz
            cv2.putText(self.canvas, f"Update: {fps} Hz", (150, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # Memory path
            cv2.putText(self.canvas, f"Memory: {self.engine.memory_path}", (300, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # Status
            status = "RUNNING" if self.running else "STOPPED"
            status_color = self.colors['low'] if self.running else self.colors['high']
            cv2.putText(self.canvas, status, (self.width - 100, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    def run(self):
        """Main dashboard loop"""
        print("="*60)
        print("COHERENCE ENGINE DASHBOARD")
        print("="*60)
        print("\nInitializing engine...")
        
        # Initialize engine
        self.init_engine()
        
        # Start engine thread
        self.running = True
        self.engine_thread = threading.Thread(target=self.run_engine)
        self.engine_thread.start()
        
        print("Dashboard running...")
        print("Press 'q' to quit")
        print("Press 's' to save snapshot")
        print("-"*60)
        
        # Create window
        window_name = "Coherence Engine Dashboard"
        
        try:
            while self.running:
                # Update display
                self.update_display()
                
                # Try to show window (might fail in headless mode)
                try:
                    cv2.imshow(window_name, self.canvas)
                    key = cv2.waitKey(50) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save snapshot
                        filename = f"coherence_snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, self.canvas)
                        print(f"Saved snapshot: {filename}")
                except:
                    # Headless mode - just save frames periodically
                    if len(self.engine.reality_history) % 50 == 0:
                        filename = "coherence_dashboard_current.jpg"
                        cv2.imwrite(filename, self.canvas)
                
                time.sleep(0.05)  # 20 FPS display update
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            # Stop engine
            self.running = False
            if self.engine_thread:
                self.engine_thread.join(timeout=2)
            
            # Shutdown engine
            if self.engine:
                self.engine.shutdown()
            
            cv2.destroyAllWindows()
            
            print("\nDashboard stopped")
            print("="*60)

if __name__ == "__main__":
    dashboard = CoherenceDashboard()
    dashboard.run()