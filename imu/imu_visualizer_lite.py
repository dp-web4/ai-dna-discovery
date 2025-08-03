#!/usr/bin/env python3
"""
Lightweight IMU Visualizer - Optimized for performance
Reduced plots and faster updates
"""
import serial
import struct
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class IMUVisualizerLite:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.running = True
        
        # Latest data
        self.data = {
            'accel': {'x': 0, 'y': 0, 'z': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0},
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'temp': 0
        }
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Performance tracking
        self.update_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.imu_rate = 0
        self.packet_count = 0
        
        # Start serial thread
        self.serial_thread = threading.Thread(target=self.serial_reader)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
    def serial_reader(self):
        """Read IMU data in background thread"""
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            buffer = b''
            last_rate_check = time.time()
            
            while self.running:
                if ser.in_waiting:
                    buffer += ser.read(ser.in_waiting)
                
                # Process packets
                while len(buffer) >= 11:
                    idx = buffer.find(b'\x55')
                    if idx == -1:
                        buffer = b''
                        break
                        
                    buffer = buffer[idx:]
                    
                    if len(buffer) >= 11:
                        packet = buffer[:11]
                        
                        # Verify checksum
                        if sum(packet[:10]) & 0xFF == packet[10]:
                            self.parse_packet(packet)
                            self.packet_count += 1
                            
                        buffer = buffer[11:]
                
                # Update IMU rate
                now = time.time()
                if now - last_rate_check > 1.0:
                    self.imu_rate = self.packet_count / (now - last_rate_check)
                    self.packet_count = 0
                    last_rate_check = now
                        
        except Exception as e:
            print(f"Serial error: {e}")
        finally:
            if 'ser' in locals():
                ser.close()
                
    def parse_packet(self, packet):
        """Parse IMU packet"""
        header2 = packet[1]
        data = packet[2:10]
        
        with self.data_lock:
            if header2 == 0x51:  # Acceleration
                self.data['accel']['x'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
                self.data['accel']['y'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
                self.data['accel']['z'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
                self.data['temp'] = struct.unpack('<h', data[6:8])[0] / 100.0
                
            elif header2 == 0x52:  # Gyroscope
                self.data['gyro']['x'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
                self.data['gyro']['y'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
                self.data['gyro']['z'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
                
            elif header2 == 0x53:  # Angle
                self.data['angle']['roll'] = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
                self.data['angle']['pitch'] = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
                self.data['angle']['yaw'] = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
                
    def create_gui(self):
        """Create lightweight GUI"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.suptitle('IMU Visualizer Lite - High Performance', fontsize=14)
        
        # Create minimal layout - 2x2 grid
        self.ax_3d = plt.subplot(2, 2, 1, projection='3d')
        self.ax_data = plt.subplot(2, 2, 2)
        self.ax_gyro = plt.subplot(2, 2, 3)
        self.ax_compass = plt.subplot(2, 2, 4, projection='polar')
        
        # Setup plots
        self.setup_3d_plot()
        self.setup_data_display()
        self.setup_gyro_display()
        self.setup_compass()
        
        # Pre-create text objects for efficiency
        self.data_text = self.ax_data.text(0.05, 0.95, '', transform=self.ax_data.transAxes,
                                          fontsize=10, verticalalignment='top', 
                                          fontfamily='monospace')
        
        # Animation - very fast update
        self.anim = FuncAnimation(self.fig, self.update_plots, interval=16, blit=True)
        
        plt.tight_layout()
        plt.show()
        
    def setup_3d_plot(self):
        """Setup simple 3D orientation display"""
        self.ax_3d.set_title('Orientation', fontsize=12)
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([-1.5, 1.5])
        self.ax_3d.set_xlabel('X', fontsize=8)
        self.ax_3d.set_ylabel('Y', fontsize=8)
        self.ax_3d.set_zlabel('Z', fontsize=8)
        
        # Pre-create line objects for axes
        self.axis_lines = {
            'x': self.ax_3d.plot([0, 1], [0, 0], [0, 0], 'r-', linewidth=3)[0],
            'y': self.ax_3d.plot([0, 0], [0, 1], [0, 0], 'g-', linewidth=3)[0],
            'z': self.ax_3d.plot([0, 0], [0, 0], [0, 1], 'b-', linewidth=3)[0]
        }
        
    def setup_data_display(self):
        """Setup data text display"""
        self.ax_data.set_title('Live Data', fontsize=12)
        self.ax_data.axis('off')
        
    def setup_gyro_display(self):
        """Setup simple gyro bar display"""
        self.ax_gyro.set_title('Gyroscope (°/s)', fontsize=12)
        self.ax_gyro.set_ylim(-500, 500)
        self.ax_gyro.set_xlim(-0.5, 2.5)
        self.ax_gyro.grid(True, alpha=0.3)
        
        # Pre-create bar objects
        self.gyro_bars = self.ax_gyro.bar([0, 1, 2], [0, 0, 0], 
                                          color=['red', 'green', 'blue'], 
                                          alpha=0.7, width=0.6)
        self.ax_gyro.set_xticks([0, 1, 2])
        self.ax_gyro.set_xticklabels(['X', 'Y', 'Z'])
        self.ax_gyro.axhline(y=0, color='white', linewidth=0.5)
        
    def setup_compass(self):
        """Setup simple compass"""
        self.ax_compass.set_title('Yaw', fontsize=12)
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        
        # Add cardinal directions
        for angle, label in [(0, 'N'), (np.pi/2, 'E'), (np.pi, 'S'), (3*np.pi/2, 'W')]:
            self.ax_compass.text(angle, 1.1, label, ha='center', va='center', fontsize=10)
            
        # Pre-create arrow
        self.compass_arrow = self.ax_compass.arrow(0, 0, 0, 0.7,
                                                  head_width=0.15, head_length=0.1,
                                                  fc='red', ec='red')
        
    def rotation_matrix(self, roll, pitch, yaw):
        """Fast rotation matrix calculation"""
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
    def update_plots(self, frame):
        """Update plots efficiently"""
        # Track FPS
        self.update_count += 1
        now = time.time()
        if now - self.last_fps_time > 1.0:
            self.fps = self.update_count / (now - self.last_fps_time)
            self.update_count = 0
            self.last_fps_time = now
        
        with self.data_lock:
            # Copy data quickly
            angles = (self.data['angle']['roll'], 
                     self.data['angle']['pitch'], 
                     self.data['angle']['yaw'])
            gyro = [self.data['gyro']['x'], 
                   self.data['gyro']['y'], 
                   self.data['gyro']['z']]
            accel = [self.data['accel']['x'],
                    self.data['accel']['y'],
                    self.data['accel']['z']]
            temp = self.data['temp']
        
        # Update 3D orientation (just axes, no box)
        R = self.rotation_matrix(*angles)
        
        # Update axis lines
        for i, axis in enumerate(['x', 'y', 'z']):
            vec = R[:, i]
            self.axis_lines[axis].set_data_3d([0, vec[0]], [0, vec[1]], [0, vec[2]])
        
        # Update data text
        text = f"Angles (°):\n"
        text += f"  Roll:  {angles[0]:7.1f}\n"
        text += f"  Pitch: {angles[1]:7.1f}\n"
        text += f"  Yaw:   {angles[2]:7.1f}\n\n"
        text += f"Accel (g): {accel[0]:5.2f}, {accel[1]:5.2f}, {accel[2]:5.2f}\n"
        text += f"Gyro (°/s): {gyro[0]:5.0f}, {gyro[1]:5.0f}, {gyro[2]:5.0f}\n\n"
        text += f"Temperature: {temp:5.1f}°C\n"
        text += f"Display FPS: {self.fps:5.1f}\n"
        text += f"IMU Rate: {self.imu_rate:5.1f} Hz"
        self.data_text.set_text(text)
        
        # Update gyro bars
        for bar, value in zip(self.gyro_bars, gyro):
            bar.set_height(value)
        
        # Update compass
        self.compass_arrow.remove()
        yaw_rad = np.radians(-angles[2] + 90)
        self.compass_arrow = self.ax_compass.arrow(0, 0, yaw_rad, 0.7,
                                                  head_width=0.15, head_length=0.1,
                                                  fc='red', ec='red')
        
        return [self.data_text] + list(self.axis_lines.values()) + list(self.gyro_bars) + [self.compass_arrow]
        
    def close(self):
        """Clean shutdown"""
        self.running = False
        if self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1)
        plt.close('all')

def main():
    print("Starting IMU Visualizer Lite...")
    print("Optimizations:")
    print("- Minimal plots for maximum performance")
    print("- 60+ FPS target")
    print("- Efficient updates with blitting")
    print("- Pre-created graphics objects")
    print("\nClose the window to exit")
    
    visualizer = IMUVisualizerLite()
    
    try:
        visualizer.create_gui()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.close()
        
if __name__ == "__main__":
    main()