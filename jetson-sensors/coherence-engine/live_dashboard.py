#!/usr/bin/env python3
"""
Live dashboard that reads from memory/experiences to show real-time coherence data.
Works alongside test_real_sensors.py without conflicting with cameras.
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from collections import deque
from datetime import datetime

class LiveDashboard:
    def __init__(self):
        # Data storage
        self.max_points = 100
        self.timestamps = deque(maxlen=self.max_points)
        self.field_values = deque(maxlen=self.max_points)
        self.sensor_values = {
            'vision': deque(maxlen=self.max_points),
            'imu': deque(maxlen=self.max_points),
            'memory': deque(maxlen=self.max_points),
            'cognition': deque(maxlen=self.max_points)
        }
        self.context_states = deque(maxlen=self.max_points)
        self.triggers = deque(maxlen=self.max_points)
        
        # Track last read position
        self.last_read_index = 0
        self.memory_dir = Path("memory/experiences")
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Create the dashboard layout."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Coherence Engine Live Dashboard', fontsize=16)
        
        # Create subplots
        # Top: Reality field over time
        self.ax_field = plt.subplot(2, 3, (1, 2))
        self.ax_field.set_title('Reality Field Over Time')
        self.ax_field.set_xlabel('Time (seconds ago)')
        self.ax_field.set_ylabel('Field Value')
        self.ax_field.set_ylim(0, 1.2)
        self.ax_field.grid(True, alpha=0.3)
        
        # Top right: Context state
        self.ax_context = plt.subplot(2, 3, 3)
        self.ax_context.set_title('Current Context')
        self.ax_context.axis('off')
        
        # Bottom left: Sensor contributions
        self.ax_sensors = plt.subplot(2, 3, 4)
        self.ax_sensors.set_title('Sensor Contributions')
        self.ax_sensors.set_ylim(0, 1.1)
        self.ax_sensors.set_ylabel('Value')
        
        # Bottom middle: Sensor history
        self.ax_history = plt.subplot(2, 3, 5)
        self.ax_history.set_title('Sensor History')
        self.ax_history.set_xlabel('Time (seconds ago)')
        self.ax_history.set_ylabel('Value')
        self.ax_history.set_ylim(0, 1.1)
        self.ax_history.grid(True, alpha=0.3)
        
        # Bottom right: Statistics
        self.ax_stats = plt.subplot(2, 3, 6)
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')
        
        # Initialize plots
        self.line_field, = self.ax_field.plot([], [], 'c-', linewidth=2, label='Reality Field')
        self.ax_field.legend()
        
        # Sensor bars
        self.bars = self.ax_sensors.bar(['Vision', 'IMU', 'Memory', 'Cognition'], 
                                        [0, 0, 0, 0],
                                        color=['green', 'red', 'blue', 'yellow'])
        
        # Sensor history lines
        self.lines_sensors = {
            'vision': self.ax_history.plot([], [], 'g-', alpha=0.7, label='Vision')[0],
            'imu': self.ax_history.plot([], [], 'r-', alpha=0.7, label='IMU')[0],
            'memory': self.ax_history.plot([], [], 'b-', alpha=0.7, label='Memory')[0],
            'cognition': self.ax_history.plot([], [], 'y-', alpha=0.7, label='Cognition')[0]
        }
        self.ax_history.legend(loc='upper right')
        
        plt.tight_layout()
        
    def load_latest_experiences(self):
        """Load new experiences from the memory directory."""
        try:
            date_str = time.strftime("%Y%m%d")
            exp_file = self.memory_dir / f"experiences_{date_str}.json"
            
            if exp_file.exists():
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                    
                # Get only new data
                if len(data) > self.last_read_index:
                    new_data = data[self.last_read_index:]
                    self.last_read_index = len(data)
                    
                    for exp in new_data:
                        # Add timestamp
                        self.timestamps.append(exp.get('timestamp', time.time()))
                        
                        # Add field value
                        self.field_values.append(exp.get('field_value', 0))
                        
                        # Add sensor readings
                        readings = exp.get('sensor_readings', {})
                        for sensor in ['vision', 'imu', 'memory', 'cognition']:
                            self.sensor_values[sensor].append(readings.get(sensor, 0))
                        
                        # Add context state
                        self.context_states.append(exp.get('context_state', 'UNKNOWN'))
                        
                        # Add trigger
                        self.triggers.append(exp.get('trigger', ''))
                        
        except Exception as e:
            print(f"Error loading experiences: {e}")
            
    def update(self, frame):
        """Update the dashboard."""
        # Load latest data
        self.load_latest_experiences()
        
        if not self.timestamps:
            return
            
        # Calculate time axis (seconds ago)
        current_time = time.time()
        time_axis = [current_time - t for t in self.timestamps]
        
        # Update reality field plot
        if len(time_axis) > 1:
            self.line_field.set_data(time_axis, self.field_values)
            self.ax_field.set_xlim(max(0, max(time_axis) - 30), max(time_axis) + 1)
            
        # Update sensor bars with latest values
        if self.sensor_values['vision']:
            latest_values = [
                self.sensor_values['vision'][-1],
                self.sensor_values['imu'][-1],
                self.sensor_values['memory'][-1],
                self.sensor_values['cognition'][-1]
            ]
            for bar, val in zip(self.bars, latest_values):
                bar.set_height(val)
                
        # Update sensor history
        if len(time_axis) > 1:
            for sensor, line in self.lines_sensors.items():
                line.set_data(time_axis, self.sensor_values[sensor])
            self.ax_history.set_xlim(max(0, max(time_axis) - 30), max(time_axis) + 1)
            
        # Update context display
        self.ax_context.clear()
        self.ax_context.axis('off')
        self.ax_context.set_title('Current Context')
        
        if self.context_states:
            current_state = self.context_states[-1]
            color = {'STABLE': 'green', 'MOVING': 'yellow', 
                    'UNSTABLE': 'orange', 'NOVEL': 'red'}.get(current_state, 'white')
            self.ax_context.text(0.5, 0.7, current_state, 
                               fontsize=24, color=color,
                               ha='center', va='center', weight='bold')
            
            if self.triggers and self.triggers[-1]:
                self.ax_context.text(0.5, 0.3, f"Trigger: {self.triggers[-1]}", 
                                   fontsize=12, color='cyan',
                                   ha='center', va='center')
                                   
        # Update statistics
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics')
        
        if self.field_values:
            recent_field = list(self.field_values)[-20:]
            stats_text = f"""
Total Experiences: {len(self.timestamps)}

Reality Field:
  Current: {self.field_values[-1]:.3f}
  Mean: {np.mean(recent_field):.3f}
  Std: {np.std(recent_field):.3f}

Latest Sensors:
  Vision: {self.sensor_values['vision'][-1]:.3f}
  IMU: {self.sensor_values['imu'][-1]:.3f}
  Memory: {self.sensor_values['memory'][-1]:.3f}
  Cognition: {self.sensor_values['cognition'][-1]:.3f}

Context Changes: {sum(1 for i in range(1, len(self.context_states)) 
                      if self.context_states[i] != self.context_states[i-1])}
"""
            self.ax_stats.text(0.1, 0.9, stats_text, 
                             fontsize=10, color='white',
                             va='top', family='monospace')
            
        plt.draw()
        
    def run(self):
        """Start the live dashboard."""
        print("Starting Live Dashboard...")
        print("Reading from memory/experiences/")
        print("Close window to exit")
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update, interval=500, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nDashboard stopped")

if __name__ == "__main__":
    dashboard = LiveDashboard()
    dashboard.run()