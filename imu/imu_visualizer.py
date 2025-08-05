#!/usr/bin/env python3
"""
Standalone IMU Visualizer with GUI
Uses matplotlib for graphical display
Can be closed without affecting the terminal session
"""
import serial
import struct
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

class IMUVisualizer:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.running = True
        self.data_queue = queue.Queue()
        
        # Latest data
        self.data = {
            'accel': {'x': 0, 'y': 0, 'z': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0},
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'mag': {'x': 0, 'y': 0, 'z': 0},
            'temp': 0,
            'timestamp': time.time()
        }
        
        # History for plots
        self.history_size = 100
        self.time_history = []
        self.accel_history = {'x': [], 'y': [], 'z': []}
        self.angle_history = {'roll': [], 'pitch': [], 'yaw': []}
        
        # Start serial thread
        self.serial_thread = threading.Thread(target=self.serial_reader)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
    def serial_reader(self):
        """Read IMU data in background thread"""
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            buffer = b''
            
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
                            
                        buffer = buffer[11:]
                        
        except Exception as e:
            print(f"Serial error: {e}")
        finally:
            if 'ser' in locals():
                ser.close()
                
    def parse_packet(self, packet):
        """Parse IMU packet"""
        header2 = packet[1]
        data = packet[2:10]
        
        if header2 == 0x51:  # Acceleration
            ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
            ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
            az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
            temp = struct.unpack('<h', data[6:8])[0] / 100.0
            self.data['accel'] = {'x': ax, 'y': ay, 'z': az}
            self.data['temp'] = temp
            
        elif header2 == 0x52:  # Gyroscope
            wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
            wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
            wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
            self.data['gyro'] = {'x': wx, 'y': wy, 'z': wz}
            
        elif header2 == 0x53:  # Angle
            roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
            pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
            yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
            self.data['angle'] = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
            
        elif header2 == 0x54:  # Magnetic
            mx = struct.unpack('<h', data[0:2])[0]
            my = struct.unpack('<h', data[2:4])[0]
            mz = struct.unpack('<h', data[4:6])[0]
            self.data['mag'] = {'x': mx, 'y': my, 'z': mz}
            
        self.data['timestamp'] = time.time()
        
    def create_gui(self):
        """Create the visualization GUI"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle('IMU Visualizer - Yahboom CMP10A', fontsize=16)
        
        # Create subplots
        # Top row: 3D orientation, Acceleration plot
        self.ax_3d = plt.subplot(2, 3, 1, projection='3d')
        self.ax_accel = plt.subplot(2, 3, 2)
        self.ax_angles = plt.subplot(2, 3, 3)
        
        # Bottom row: Text data, Gyro visualization, Compass
        self.ax_text = plt.subplot(2, 3, 4)
        self.ax_gyro = plt.subplot(2, 3, 5, projection='polar')
        self.ax_compass = plt.subplot(2, 3, 6, projection='polar')
        
        # Setup axes
        self.setup_3d_plot()
        self.setup_accel_plot()
        self.setup_angle_plot()
        self.setup_text_display()
        self.setup_gyro_display()
        self.setup_compass()
        
        # Animation
        self.anim = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)
        
        plt.tight_layout()
        plt.show()
        
    def setup_3d_plot(self):
        """Setup 3D orientation display"""
        self.ax_3d.set_title('3D Orientation')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        
        # Create a box to represent orientation
        self.box_points = np.array([
            [-1, -0.5, -0.2], [1, -0.5, -0.2], [1, 0.5, -0.2], [-1, 0.5, -0.2],  # bottom
            [-1, -0.5, 0.2], [1, -0.5, 0.2], [1, 0.5, 0.2], [-1, 0.5, 0.2]   # top
        ])
        
    def setup_accel_plot(self):
        """Setup acceleration plot"""
        self.ax_accel.set_title('Acceleration (g)')
        self.ax_accel.set_xlabel('Time')
        self.ax_accel.set_ylabel('Acceleration')
        self.ax_accel.set_ylim([-2, 2])
        self.ax_accel.grid(True, alpha=0.3)
        
        self.accel_lines = {
            'x': self.ax_accel.plot([], [], 'r-', label='X')[0],
            'y': self.ax_accel.plot([], [], 'g-', label='Y')[0],
            'z': self.ax_accel.plot([], [], 'b-', label='Z')[0]
        }
        self.ax_accel.legend(loc='upper right')
        
    def setup_angle_plot(self):
        """Setup angle plot"""
        self.ax_angles.set_title('Euler Angles (°)')
        self.ax_angles.set_xlabel('Time')
        self.ax_angles.set_ylabel('Angle')
        self.ax_angles.set_ylim([-180, 180])
        self.ax_angles.grid(True, alpha=0.3)
        
        self.angle_lines = {
            'roll': self.ax_angles.plot([], [], 'r-', label='Roll')[0],
            'pitch': self.ax_angles.plot([], [], 'g-', label='Pitch')[0],
            'yaw': self.ax_angles.plot([], [], 'b-', label='Yaw')[0]
        }
        self.ax_angles.legend(loc='upper right')
        
    def setup_text_display(self):
        """Setup text data display"""
        self.ax_text.set_title('Current Values')
        self.ax_text.axis('off')
        self.text_display = self.ax_text.text(0.05, 0.95, '', transform=self.ax_text.transAxes,
                                              fontsize=10, verticalalignment='top', 
                                              fontfamily='monospace')
        
    def setup_gyro_display(self):
        """Setup gyroscope display"""
        self.ax_gyro.set_title('Gyroscope (°/s)')
        self.ax_gyro.set_ylim(0, 2000)
        self.gyro_bars = []
        
    def setup_compass(self):
        """Setup compass display"""
        self.ax_compass.set_title('Compass (Yaw)')
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        self.compass_arrow = self.ax_compass.arrow(0, 0, 0, 0.8, 
                                                   head_width=0.1, head_length=0.1,
                                                   fc='red', ec='red')
        
    def rotation_matrix(self, roll, pitch, yaw):
        """Create rotation matrix from Euler angles"""
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
                      
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
                      
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
                      
        return Rz @ Ry @ Rx
        
    def update_plots(self, frame):
        """Update all plots"""
        # Update history
        current_time = time.time()
        self.time_history.append(current_time)
        
        for axis in ['x', 'y', 'z']:
            self.accel_history[axis].append(self.data['accel'][axis])
            
        for axis in ['roll', 'pitch', 'yaw']:
            self.angle_history[axis].append(self.data['angle'][axis])
            
        # Keep only recent history
        if len(self.time_history) > self.history_size:
            self.time_history = self.time_history[-self.history_size:]
            for axis in ['x', 'y', 'z']:
                self.accel_history[axis] = self.accel_history[axis][-self.history_size:]
            for axis in ['roll', 'pitch', 'yaw']:
                self.angle_history[axis] = self.angle_history[axis][-self.history_size:]
                
        # Update 3D orientation
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Orientation')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        
        # Rotate box
        R = self.rotation_matrix(self.data['angle']['roll'], 
                               self.data['angle']['pitch'],
                               self.data['angle']['yaw'])
        rotated_points = (R @ self.box_points.T).T
        
        # Draw box
        # Bottom face
        bottom = [0, 1, 2, 3, 0]
        self.ax_3d.plot(rotated_points[bottom, 0], 
                       rotated_points[bottom, 1],
                       rotated_points[bottom, 2], 'b-')
        # Top face
        top = [4, 5, 6, 7, 4]
        self.ax_3d.plot(rotated_points[top, 0],
                       rotated_points[top, 1], 
                       rotated_points[top, 2], 'b-')
        # Vertical edges
        for i in range(4):
            self.ax_3d.plot([rotated_points[i, 0], rotated_points[i+4, 0]],
                           [rotated_points[i, 1], rotated_points[i+4, 1]],
                           [rotated_points[i, 2], rotated_points[i+4, 2]], 'b-')
                           
        # Update acceleration plot
        if self.time_history:
            time_array = np.array(self.time_history) - self.time_history[0]
            for axis in ['x', 'y', 'z']:
                self.accel_lines[axis].set_data(time_array, self.accel_history[axis])
            self.ax_accel.set_xlim(max(0, time_array[-1] - 10), time_array[-1])
            
        # Update angle plot
        if self.time_history:
            for axis in ['roll', 'pitch', 'yaw']:
                self.angle_lines[axis].set_data(time_array, self.angle_history[axis])
            self.ax_angles.set_xlim(max(0, time_array[-1] - 10), time_array[-1])
            
        # Update text display
        text = f"Acceleration (g):\n"
        text += f"  X: {self.data['accel']['x']:7.3f}\n"
        text += f"  Y: {self.data['accel']['y']:7.3f}\n"
        text += f"  Z: {self.data['accel']['z']:7.3f}\n"
        text += f"  Mag: {np.sqrt(sum(v**2 for v in self.data['accel'].values())):7.3f}\n\n"
        text += f"Gyroscope (°/s):\n"
        text += f"  X: {self.data['gyro']['x']:7.1f}\n"
        text += f"  Y: {self.data['gyro']['y']:7.1f}\n"
        text += f"  Z: {self.data['gyro']['z']:7.1f}\n\n"
        text += f"Temperature: {self.data.get('temp', 0):5.1f}°C"
        self.text_display.set_text(text)
        
        # Update gyro display
        self.ax_gyro.clear()
        self.ax_gyro.set_title('Gyroscope (°/s)')
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        values = [abs(self.data['gyro']['x']), 
                 abs(self.data['gyro']['y']),
                 abs(self.data['gyro']['z'])]
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for angle, value, color, label in zip(angles, values, colors, labels):
            self.ax_gyro.bar(angle, value, width=0.5, bottom=0,
                           color=color, alpha=0.7, label=label)
        self.ax_gyro.set_ylim(0, 2000)
        self.ax_gyro.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        # Update compass
        self.ax_compass.clear()
        self.ax_compass.set_title('Compass (Yaw)')
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        yaw_rad = np.radians(self.data['angle']['yaw'])
        self.ax_compass.arrow(0, 0, yaw_rad, 0.8,
                            head_width=0.15, head_length=0.1,
                            fc='red', ec='red')
        # Add cardinal directions
        self.ax_compass.text(0, 1.1, 'N', ha='center', va='center')
        self.ax_compass.text(np.pi/2, 1.1, 'E', ha='center', va='center')
        self.ax_compass.text(np.pi, 1.1, 'S', ha='center', va='center')
        self.ax_compass.text(3*np.pi/2, 1.1, 'W', ha='center', va='center')
        
    def close(self):
        """Clean shutdown"""
        self.running = False
        if self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1)
        plt.close('all')

def main():
    print("Starting IMU Visualizer...")
    print("Close the window to exit cleanly")
    
    visualizer = IMUVisualizer()
    
    try:
        visualizer.create_gui()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.close()
        
if __name__ == "__main__":
    main()