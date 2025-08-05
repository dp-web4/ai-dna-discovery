#!/usr/bin/env python3
"""
IMU Orientation Mapper
Helps configure IMU-to-camera coordinate system mapping
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import json
import os

@dataclass
class OrientationConfig:
    """Configuration for IMU orientation mapping"""
    # Axis mapping: which IMU axis corresponds to which camera axis
    axis_map: Dict[str, str] = None  # e.g., {'x': 'y', 'y': '-x', 'z': 'z'}
    # Rotation offset in degrees
    roll_offset: float = 0.0
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    # Sign flips for each axis
    flip_roll: bool = False
    flip_pitch: bool = False
    flip_yaw: bool = False
    
    def __post_init__(self):
        if self.axis_map is None:
            # Default: IMU and camera have same orientation
            self.axis_map = {'x': 'x', 'y': 'y', 'z': 'z'}
    
    def to_dict(self):
        return {
            'axis_map': self.axis_map,
            'roll_offset': self.roll_offset,
            'pitch_offset': self.pitch_offset,
            'yaw_offset': self.yaw_offset,
            'flip_roll': self.flip_roll,
            'flip_pitch': self.flip_pitch,
            'flip_yaw': self.flip_yaw
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def save(self, filename='imu_orientation_config.json'):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved orientation config to {filename}")
    
    @classmethod
    def load(cls, filename='imu_orientation_config.json'):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return cls.from_dict(json.load(f))
        return cls()  # Return default if no config exists

class IMUOrientationMapper:
    """Maps IMU coordinates to camera/world coordinates"""
    
    def __init__(self, config: OrientationConfig = None):
        self.config = config or OrientationConfig()
        
    def map_angles(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
        """Map IMU angles to camera coordinate system"""
        # Apply axis mapping
        angles = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        
        # Apply sign flips
        if self.config.flip_roll:
            angles['roll'] = -angles['roll']
        if self.config.flip_pitch:
            angles['pitch'] = -angles['pitch']
        if self.config.flip_yaw:
            angles['yaw'] = -angles['yaw']
            
        # Apply offsets
        angles['roll'] += self.config.roll_offset
        angles['pitch'] += self.config.pitch_offset
        angles['yaw'] += self.config.yaw_offset
        
        # Normalize to -180 to 180
        for key in angles:
            angles[key] = ((angles[key] + 180) % 360) - 180
            
        return angles['roll'], angles['pitch'], angles['yaw']
    
    def map_vector(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Map IMU vector (accel/gyro) to camera coordinates"""
        vec = {'x': x, 'y': y, 'z': z}
        result = {}
        
        for cam_axis, imu_axis in self.config.axis_map.items():
            sign = 1
            if imu_axis.startswith('-'):
                sign = -1
                imu_axis = imu_axis[1:]
            result[cam_axis] = sign * vec[imu_axis]
            
        return result['x'], result['y'], result['z']
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix for the configured mapping"""
        matrix = np.zeros((3, 3))
        axis_indices = {'x': 0, 'y': 1, 'z': 2}
        
        for cam_axis, imu_axis in self.config.axis_map.items():
            sign = 1
            if imu_axis.startswith('-'):
                sign = -1
                imu_axis = imu_axis[1:]
            
            cam_idx = axis_indices[cam_axis]
            imu_idx = axis_indices[imu_axis]
            matrix[cam_idx, imu_idx] = sign
            
        return matrix

def create_common_configs():
    """Create common IMU mounting configurations"""
    configs = {}
    
    # IMU mounted upside down
    configs['upside_down'] = OrientationConfig(
        axis_map={'x': 'x', 'y': '-y', 'z': '-z'},
        flip_roll=True
    )
    
    # IMU rotated 90° around Z axis (IMU X points to camera Y)
    configs['rotated_90_z'] = OrientationConfig(
        axis_map={'x': 'y', 'y': '-x', 'z': 'z'},
        yaw_offset=90
    )
    
    # IMU rotated 90° around X axis (IMU Y points up)
    configs['rotated_90_x'] = OrientationConfig(
        axis_map={'x': 'x', 'y': '-z', 'z': 'y'},
        roll_offset=90
    )
    
    # IMU mounted on side (common for compact mounting)
    configs['side_mount'] = OrientationConfig(
        axis_map={'x': 'z', 'y': 'y', 'z': '-x'},
        pitch_offset=90
    )
    
    return configs

def interactive_setup():
    """Interactive setup to determine IMU orientation"""
    print("\nIMU Orientation Setup")
    print("=" * 50)
    print("\nThis tool helps map IMU coordinates to camera coordinates")
    print("\nCommon mounting configurations:")
    
    configs = create_common_configs()
    options = ['custom'] + list(configs.keys())
    
    for i, option in enumerate(options):
        print(f"{i}: {option}")
    
    try:
        choice = int(input("\nSelect configuration (0 for custom): "))
        
        if choice == 0:
            # Custom configuration
            config = OrientationConfig()
            
            print("\nFor each camera axis, specify which IMU axis it corresponds to")
            print("Use '-' prefix to flip the axis (e.g., '-x' means negative X)")
            
            for axis in ['x', 'y', 'z']:
                while True:
                    imu_axis = input(f"Camera {axis.upper()} axis maps to IMU axis: ").lower().strip()
                    if imu_axis in ['x', 'y', 'z', '-x', '-y', '-z']:
                        config.axis_map[axis] = imu_axis
                        break
                    print("Invalid input. Use x, y, z, -x, -y, or -z")
            
            print("\nEnter rotation offsets (degrees):")
            config.roll_offset = float(input("Roll offset [0]: ") or 0)
            config.pitch_offset = float(input("Pitch offset [0]: ") or 0)
            config.yaw_offset = float(input("Yaw offset [0]: ") or 0)
            
        else:
            config_name = options[choice]
            config = configs[config_name]
            print(f"\nUsing {config_name} configuration")
        
        # Save configuration
        config.save()
        
        # Show the rotation matrix
        mapper = IMUOrientationMapper(config)
        print("\nRotation matrix:")
        print(mapper.get_rotation_matrix())
        
        print("\nConfiguration saved to imu_orientation_config.json")
        print("Use this file with the stabilized vision system")
        
    except (ValueError, IndexError):
        print("Invalid selection")

def test_mapping():
    """Test the current mapping configuration"""
    config = OrientationConfig.load()
    mapper = IMUOrientationMapper(config)
    
    print("\nTesting current configuration:")
    print(f"Config: {config.to_dict()}")
    
    # Test angle mapping
    test_angles = [(90, 0, 0), (0, 90, 0), (0, 0, 90)]
    print("\nAngle mapping tests:")
    for roll, pitch, yaw in test_angles:
        mapped = mapper.map_angles(roll, pitch, yaw)
        print(f"IMU ({roll}, {pitch}, {yaw}) → Camera {mapped}")
    
    # Test vector mapping
    test_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    print("\nVector mapping tests:")
    for x, y, z in test_vectors:
        mapped = mapper.map_vector(x, y, z)
        print(f"IMU ({x}, {y}, {z}) → Camera {mapped}")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_mapping()
    else:
        interactive_setup()

if __name__ == "__main__":
    main()