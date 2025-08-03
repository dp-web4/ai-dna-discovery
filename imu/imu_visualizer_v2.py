#!/usr/bin/env python3
"""
Improved IMU Visualizer with better performance and gyro display
"""
import serial
import struct
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import datetime

class IMUVisualizerV2:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.running = True
        
        # Latest data with timestamps
        self.data = {
            'accel': {'x': 0, 'y': 0, 'z': 0, 'timestamp': 0},
            'gyro': {'x': 0, 'y': 0, 'z': 0, 'timestamp': 0},
            'angle': {'roll': 0, 'pitch': 0, 'yaw': 0, 'timestamp': 0},
            'mag': {'x': 0, 'y': 0, 'z': 0, 'timestamp': 0},
            'temp': 0
        }
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # History for plots (using deque for efficiency)
        self.history_size = 200
        self.time_history = deque(maxlen=self.history_size)
        self.accel_history = {'x': deque(maxlen=self.history_size), 
                             'y': deque(maxlen=self.history_size), 
                             'z': deque(maxlen=self.history_size)}
        self.gyro_history = {'x': deque(maxlen=self.history_size),
                            'y': deque(maxlen=self.history_size),
                            'z': deque(maxlen=self.history_size)}
        self.angle_history = {'roll': deque(maxlen=self.history_size),
                             'pitch': deque(maxlen=self.history_size),
                             'yaw': deque(maxlen=self.history_size)}
        
        # Performance tracking
        self.packet_count = 0
        self.last_packet_time = time.time()
        self.update_rate = 0
        
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
                        checksum = sum(packet[:10]) & 0xFF
                        if checksum == packet[10]:
                            self.parse_packet(packet)
                            self.packet_count += 1
                            
                        buffer = buffer[11:]
                        
                # Calculate update rate
                current_time = time.time()
                if current_time - self.last_packet_time > 1.0:
                    self.update_rate = self.packet_count / (current_time - self.last_packet_time)
                    self.packet_count = 0
                    self.last_packet_time = current_time
                        
        except Exception as e:
            print(f"Serial error: {e}")
        finally:
            if 'ser' in locals():
                ser.close()
                
    def parse_packet(self, packet):
        """Parse IMU packet with proper timestamp"""
        header2 = packet[1]
        data = packet[2:10]
        timestamp = time.time()
        
        with self.data_lock:
            if header2 == 0x51:  # Acceleration
                ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
                ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
                az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
                temp = struct.unpack('<h', data[6:8])[0] / 100.0
                self.data['accel'] = {'x': ax, 'y': ay, 'z': az, 'timestamp': timestamp}
                self.data['temp'] = temp
                
                # Update history
                self.time_history.append(timestamp)
                self.accel_history['x'].append(ax)
                self.accel_history['y'].append(ay)
                self.accel_history['z'].append(az)
                
            elif header2 == 0x52:  # Gyroscope
                wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
                wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
                wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000
                self.data['gyro'] = {'x': wx, 'y': wy, 'z': wz, 'timestamp': timestamp}
                
                # Update gyro history
                if len(self.time_history) > 0:
                    self.gyro_history['x'].append(wx)
                    self.gyro_history['y'].append(wy)
                    self.gyro_history['z'].append(wz)
                
            elif header2 == 0x53:  # Angle
                roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
                pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
                yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
                self.data['angle'] = {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'timestamp': timestamp}
                
                # Update angle history
                if len(self.time_history) > 0:
                    self.angle_history['roll'].append(roll)
                    self.angle_history['pitch'].append(pitch)
                    self.angle_history['yaw'].append(yaw)
                
            elif header2 == 0x54:  # Magnetic
                mx = struct.unpack('<h', data[0:2])[0]
                my = struct.unpack('<h', data[2:4])[0]
                mz = struct.unpack('<h', data[4:6])[0]
                self.data['mag'] = {'x': mx, 'y': my, 'z': mz, 'timestamp': timestamp}
                
    def create_gui(self):
        """Create the visualization GUI with better layout"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('IMU Visualizer V2 - Yahboom CMP10A', fontsize=16)
        
        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Top row: 3D orientation (larger), Text data
        self.ax_3d = self.fig.add_subplot(gs[0:2, 0], projection='3d')
        self.ax_text = self.fig.add_subplot(gs[0, 1])
        self.ax_status = self.fig.add_subplot(gs[1, 1])
        
        # Middle: Acceleration and Gyro plots
        self.ax_accel = self.fig.add_subplot(gs[0, 2])
        self.ax_gyro = self.fig.add_subplot(gs[1, 2])
        
        # Bottom row: Angles, Gyro bars, Compass
        self.ax_angles = self.fig.add_subplot(gs[2, 0])
        self.ax_gyro_bars = self.fig.add_subplot(gs[2, 1])
        self.ax_compass = self.fig.add_subplot(gs[2, 2], projection='polar')
        
        # Setup all plots
        self.setup_3d_plot()
        self.setup_text_display()
        self.setup_status_display()
        self.setup_accel_plot()
        self.setup_gyro_plot()
        self.setup_angle_plot()
        self.setup_gyro_bars()
        self.setup_compass()
        
        # Animation - faster update rate
        self.anim = FuncAnimation(self.fig, self.update_plots, interval=20, blit=False)
        
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
        
        # Create a more detailed box
        self.box_vertices = np.array([
            [-1, -0.5, -0.2], [1, -0.5, -0.2], [1, 0.5, -0.2], [-1, 0.5, -0.2],  # bottom
            [-1, -0.5, 0.2], [1, -0.5, 0.2], [1, 0.5, 0.2], [-1, 0.5, 0.2]   # top
        ])
        
        # Add reference axes
        self.ref_axes_length = 1.5
        
    def setup_text_display(self):
        """Setup text data display"""
        self.ax_text.set_title('Current Values')
        self.ax_text.axis('off')
        self.text_display = self.ax_text.text(0.05, 0.95, '', transform=self.ax_text.transAxes,
                                              fontsize=9, verticalalignment='top', 
                                              fontfamily='monospace')
        
    def setup_status_display(self):
        """Setup status display"""
        self.ax_status.set_title('System Status')
        self.ax_status.axis('off')
        self.status_display = self.ax_status.text(0.05, 0.95, '', transform=self.ax_status.transAxes,
                                                 fontsize=9, verticalalignment='top',
                                                 fontfamily='monospace')
        
    def setup_accel_plot(self):
        """Setup acceleration plot"""
        self.ax_accel.set_title('Acceleration (g)')
        self.ax_accel.set_xlabel('Time (s)')
        self.ax_accel.set_ylabel('Acceleration')
        self.ax_accel.set_ylim([-2, 2])
        self.ax_accel.grid(True, alpha=0.3)
        
        self.accel_lines = {
            'x': self.ax_accel.plot([], [], 'r-', label='X', linewidth=1)[0],
            'y': self.ax_accel.plot([], [], 'g-', label='Y', linewidth=1)[0],
            'z': self.ax_accel.plot([], [], 'b-', label='Z', linewidth=1)[0]
        }
        self.ax_accel.legend(loc='upper right')
        
    def setup_gyro_plot(self):
        """Setup gyroscope time series plot"""
        self.ax_gyro.set_title('Gyroscope (°/s)')
        self.ax_gyro.set_xlabel('Time (s)')
        self.ax_gyro.set_ylabel('Angular Velocity')
        self.ax_gyro.set_ylim([-100, 100])
        self.ax_gyro.grid(True, alpha=0.3)
        
        self.gyro_lines = {
            'x': self.ax_gyro.plot([], [], 'r-', label='X', linewidth=1)[0],
            'y': self.ax_gyro.plot([], [], 'g-', label='Y', linewidth=1)[0],
            'z': self.ax_gyro.plot([], [], 'b-', label='Z', linewidth=1)[0]
        }
        self.ax_gyro.legend(loc='upper right')
        
    def setup_angle_plot(self):
        """Setup angle plot"""
        self.ax_angles.set_title('Euler Angles (°)')
        self.ax_angles.set_xlabel('Time (s)')
        self.ax_angles.set_ylabel('Angle')
        self.ax_angles.set_ylim([-180, 180])
        self.ax_angles.grid(True, alpha=0.3)
        
        self.angle_lines = {
            'roll': self.ax_angles.plot([], [], 'r-', label='Roll', linewidth=1)[0],
            'pitch': self.ax_angles.plot([], [], 'g-', label='Pitch', linewidth=1)[0],
            'yaw': self.ax_angles.plot([], [], 'b-', label='Yaw', linewidth=1)[0]
        }
        self.ax_angles.legend(loc='upper right')
        
    def setup_gyro_bars(self):
        """Setup gyroscope bar display"""
        self.ax_gyro_bars.set_title('Gyroscope Magnitude')
        self.ax_gyro_bars.set_ylim(-500, 500)
        self.ax_gyro_bars.set_xlim(-0.5, 2.5)
        self.ax_gyro_bars.grid(True, alpha=0.3)
        
    def setup_compass(self):
        """Setup compass display"""
        self.ax_compass.set_title('Compass (Yaw)')
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        
    def rotation_matrix(self, roll, pitch, yaw):
        """Create rotation matrix from Euler angles"""
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
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
        """Update all plots with thread-safe data access"""
        with self.data_lock:
            # Copy current data
            current_data = {
                'accel': self.data['accel'].copy(),
                'gyro': self.data['gyro'].copy(),
                'angle': self.data['angle'].copy(),
                'mag': self.data['mag'].copy(),
                'temp': self.data['temp']
            }
            
            # Copy history data
            if len(self.time_history) > 0:
                time_array = np.array(self.time_history) - self.time_history[0]
                accel_data = {k: np.array(v) for k, v in self.accel_history.items()}
                gyro_data = {k: np.array(v) for k, v in self.gyro_history.items()}
                angle_data = {k: np.array(v) for k, v in self.angle_history.items()}
            else:
                time_array = np.array([])
                accel_data = gyro_data = angle_data = None
                
        # Update 3D orientation
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Orientation')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        
        # Draw reference axes (fixed)
        self.ax_3d.plot([0, self.ref_axes_length], [0, 0], [0, 0], 'r--', alpha=0.3, label='X ref')
        self.ax_3d.plot([0, 0], [0, self.ref_axes_length], [0, 0], 'g--', alpha=0.3, label='Y ref')
        self.ax_3d.plot([0, 0], [0, 0], [0, self.ref_axes_length], 'b--', alpha=0.3, label='Z ref')
        
        # Rotate and draw box
        R = self.rotation_matrix(current_data['angle']['roll'], 
                               current_data['angle']['pitch'],
                               current_data['angle']['yaw'])
        rotated_points = (R @ self.box_vertices.T).T
        
        # Draw box faces
        faces = [
            [0, 1, 2, 3, 0],  # bottom
            [4, 5, 6, 7, 4],  # top
            [0, 1, 5, 4, 0],  # front
            [2, 3, 7, 6, 2],  # back
            [0, 3, 7, 4, 0],  # left
            [1, 2, 6, 5, 1]   # right
        ]
        
        for face in faces:
            self.ax_3d.plot(rotated_points[face, 0], 
                           rotated_points[face, 1],
                           rotated_points[face, 2], 'b-', linewidth=1.5)
            
        # Draw oriented axes
        axes_length = 1.2
        x_axis = R @ np.array([axes_length, 0, 0])
        y_axis = R @ np.array([0, axes_length, 0])
        z_axis = R @ np.array([0, 0, axes_length])
        
        self.ax_3d.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], 'r-', linewidth=3, label='X')
        self.ax_3d.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], 'g-', linewidth=3, label='Y')
        self.ax_3d.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], 'b-', linewidth=3, label='Z')
        
        # Update acceleration plot
        if time_array.size > 0 and accel_data:
            for axis in ['x', 'y', 'z']:
                self.accel_lines[axis].set_data(time_array, accel_data[axis])
            self.ax_accel.set_xlim(max(0, time_array[-1] - 10), time_array[-1] + 0.5)
            
        # Update gyro plot
        if time_array.size > 0 and gyro_data and len(gyro_data['x']) > 0:
            # Ensure gyro data matches time array length
            min_len = min(len(time_array), len(gyro_data['x']))
            if min_len > 0:
                for axis in ['x', 'y', 'z']:
                    self.gyro_lines[axis].set_data(time_array[:min_len], gyro_data[axis][:min_len])
                self.ax_gyro.set_xlim(max(0, time_array[-1] - 10), time_array[-1] + 0.5)
            
        # Update angle plot
        if time_array.size > 0 and angle_data and len(angle_data['roll']) > 0:
            min_len = min(len(time_array), len(angle_data['roll']))
            if min_len > 0:
                for axis in ['roll', 'pitch', 'yaw']:
                    self.angle_lines[axis].set_data(time_array[:min_len], angle_data[axis][:min_len])
                self.ax_angles.set_xlim(max(0, time_array[-1] - 10), time_array[-1] + 0.5)
            
        # Update text display
        text = f"Acceleration (g):\n"
        text += f"  X: {current_data['accel']['x']:7.3f}\n"
        text += f"  Y: {current_data['accel']['y']:7.3f}\n"
        text += f"  Z: {current_data['accel']['z']:7.3f}\n"
        text += f"  Mag: {np.sqrt(sum(v**2 for v in [current_data['accel']['x'], current_data['accel']['y'], current_data['accel']['z']])):7.3f}\n\n"
        text += f"Gyroscope (°/s):\n"
        text += f"  X: {current_data['gyro']['x']:7.1f}\n"
        text += f"  Y: {current_data['gyro']['y']:7.1f}\n"
        text += f"  Z: {current_data['gyro']['z']:7.1f}\n\n"
        text += f"Angles (°):\n"
        text += f"  Roll:  {current_data['angle']['roll']:7.1f}\n"
        text += f"  Pitch: {current_data['angle']['pitch']:7.1f}\n"
        text += f"  Yaw:   {current_data['angle']['yaw']:7.1f}"
        self.text_display.set_text(text)
        
        # Update status display
        status = f"Port: {self.port}\n"
        status += f"Baud: {self.baud}\n"
        status += f"Temperature: {current_data['temp']:5.1f}°C\n\n"
        status += f"Update Rate: {self.update_rate:.1f} Hz\n"
        status += f"Frame Rate: {1000/20:.1f} FPS\n\n"
        
        # Data freshness
        now = time.time()
        accel_age = now - current_data['accel']['timestamp'] if current_data['accel']['timestamp'] > 0 else 999
        gyro_age = now - current_data['gyro']['timestamp'] if current_data['gyro']['timestamp'] > 0 else 999
        angle_age = now - current_data['angle']['timestamp'] if current_data['angle']['timestamp'] > 0 else 999
        
        status += f"Data Age (ms):\n"
        status += f"  Accel: {accel_age*1000:4.0f}\n"
        status += f"  Gyro:  {gyro_age*1000:4.0f}\n"
        status += f"  Angle: {angle_age*1000:4.0f}"
        self.status_display.set_text(status)
        
        # Update gyro bars
        self.ax_gyro_bars.clear()
        self.ax_gyro_bars.set_title('Gyroscope Magnitude')
        self.ax_gyro_bars.set_ylim(-500, 500)
        self.ax_gyro_bars.set_xlim(-0.5, 2.5)
        self.ax_gyro_bars.grid(True, alpha=0.3)
        
        positions = [0, 1, 2]
        values = [current_data['gyro']['x'], current_data['gyro']['y'], current_data['gyro']['z']]
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        bars = self.ax_gyro_bars.bar(positions, values, color=colors, alpha=0.7, width=0.6)
        self.ax_gyro_bars.set_xticks(positions)
        self.ax_gyro_bars.set_xticklabels(labels)
        self.ax_gyro_bars.axhline(y=0, color='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax_gyro_bars.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{value:.0f}', ha='center', va='bottom' if height > 0 else 'top',
                                  fontsize=8)
        
        # Update compass
        self.ax_compass.clear()
        self.ax_compass.set_title('Compass (Yaw)')
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        
        # Draw compass rose
        for angle, label in [(0, 'N'), (np.pi/2, 'E'), (np.pi, 'S'), (3*np.pi/2, 'W')]:
            self.ax_compass.text(angle, 1.1, label, ha='center', va='center', fontsize=12, weight='bold')
            
        # Draw degree markings
        for deg in range(0, 360, 30):
            angle = np.radians(deg)
            self.ax_compass.plot([angle, angle], [0.9, 0.95], 'w-', linewidth=1)
            if deg % 90 != 0:
                self.ax_compass.text(angle, 1.05, str(deg), ha='center', va='center', fontsize=8)
        
        # Draw yaw arrow
        yaw_rad = np.radians(-current_data['angle']['yaw'] + 90)  # Convert to compass bearing
        self.ax_compass.arrow(0, 0, yaw_rad, 0.7,
                            head_width=0.15, head_length=0.1,
                            fc='red', ec='red', linewidth=2)
        
        # Add yaw value in center
        self.ax_compass.text(0, 0, f"{current_data['angle']['yaw']:.0f}°", 
                           ha='center', va='center', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
    def close(self):
        """Clean shutdown"""
        self.running = False
        if self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1)
        plt.close('all')

def main():
    print("Starting IMU Visualizer V2...")
    print("Improvements:")
    print("- Better gyroscope display with time series and bar chart")
    print("- Higher frame rate (50 FPS)")
    print("- Data freshness indicators")
    print("- Improved 3D visualization")
    print("\nClose the window to exit cleanly")
    
    visualizer = IMUVisualizerV2()
    
    try:
        visualizer.create_gui()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.close()
        
if __name__ == "__main__":
    main()