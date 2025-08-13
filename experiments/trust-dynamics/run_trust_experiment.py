#!/usr/bin/env python3
"""
Trust Dynamics Experiment Runner
Tests how trust weights adapt to sensor conflicts
August 12, 2025
"""

import sys
import time
import signal
import threading
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "coherence-engine/plugins/common"))
sys.path.append(str(Path(__file__).parent.parent.parent / "coherence-engine/plugins/jetson"))

# Import the working coherence engine
from run_jetson import CoherenceWithVideo

# Import our logging effector
from logging_effector import LoggingEffector

class TrustDynamicsExperiment:
    """Orchestrates trust dynamics experiment with phase transitions"""
    
    def __init__(self):
        self.coherence = CoherenceWithVideo()
        self.logger = LoggingEffector(
            log_dir="experiments/trust-dynamics",
            log_rate=10.0  # 10 Hz logging
        )
        
        self.phases = [
            ("baseline", 30, "Normal operation - all sensors working"),
            ("left_occlusion", 30, "Left camera covered"),
            ("motion_conflict", 30, "Static scene but moving IMU"),
            ("recovery", 30, "Return to normal"),
            ("full_occlusion", 30, "Both cameras covered")
        ]
        
        self.current_phase_idx = 0
        self.phase_start_time = None
        self.experiment_running = True
        
    def run_experiment(self):
        """Main experiment loop"""
        print("\n" + "="*60)
        print("TRUST DYNAMICS EXPERIMENT")
        print("="*60)
        
        # Initialize coherence engine
        self.coherence.initialize()
        
        # Start phase management thread
        phase_thread = threading.Thread(target=self.manage_phases)
        phase_thread.daemon = True
        phase_thread.start()
        
        # Override the main loop to add logging
        self.run_with_logging()
        
    def manage_phases(self):
        """Background thread to manage experiment phases"""
        for phase_name, duration, description in self.phases:
            if not self.experiment_running:
                break
                
            print(f"\n{'='*50}")
            print(f"PHASE: {phase_name.upper()}")
            print(f"Duration: {duration}s")
            print(f"Action: {description}")
            print(f"{'='*50}\n")
            
            # Update logger phase
            self.logger.set_phase(phase_name)
            
            # Give user instructions
            if phase_name == "left_occlusion":
                print(">>> COVER THE LEFT CAMERA NOW <<<")
            elif phase_name == "motion_conflict":
                print(">>> UNCOVER CAMERA, THEN SHAKE THE DEVICE <<<")
            elif phase_name == "recovery":
                print(">>> STOP SHAKING, RETURN TO NORMAL <<<")
            elif phase_name == "full_occlusion":
                print(">>> COVER BOTH CAMERAS NOW <<<")
                
            # Wait for phase duration
            time.sleep(duration)
            
        print("\n>>> EXPERIMENT COMPLETE - Press 'q' to exit <<<")
        
    def run_with_logging(self):
        """Modified main loop with logging effector"""
        print("\nStarting coherence engine with logging...")
        
        while self.coherence.running and self.experiment_running:
            # Read camera frames
            ret_l, frame_l = self.coherence.cap_l.read()
            ret_r, frame_r = self.coherence.cap_r.read()
            
            if not ret_l or not ret_r:
                continue
                
            # Compute sensor data
            self.coherence.camera_motion = self.coherence.compute_camera_motion(frame_l, frame_r)
            
            # Update IMU data
            if not self.coherence.use_real_imu:
                self.coherence.simulate_imu()
                
            # Calculate stability from IMU
            import numpy as np
            gyro_mag = np.linalg.norm(self.coherence.imu_data["gyroscope"])
            self.coherence.imu_stability = 1.0 / (1.0 + gyro_mag * 10)
            
            # Update coherence engine
            self.coherence.update_context()
            self.coherence.update_weights()
            self.coherence.update_trust()
            self.coherence.compute_reality_field()
            
            # Create context for logging
            context = {
                "tick": self.coherence.tick_count,
                "state": self.coherence.context_state.name,
                "trust_weights": self.coherence.trust_weights.copy(),
                "camera_motion": self.coherence.camera_motion,
                "imu_stability": self.coherence.imu_stability,
                "imu_data": self.coherence.imu_data.copy()
            }
            
            # Log via effector
            self.logger.effect(self.coherence.reality_field, context)
            
            # Create and display dashboard
            dashboard = self.coherence.create_dashboard(frame_l, frame_r)
            import cv2
            cv2.imshow(self.coherence.window_name, dashboard)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.experiment_running = False
                break
            elif key == ord('s'):
                filename = f"trust_experiment_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, dashboard)
                print(f"Screenshot saved: {filename}")
                
            # Update FPS and tick
            current_time = time.time()
            fps = 1.0 / (current_time - self.coherence.last_time + 0.001)
            self.coherence.fps_history.append(fps)
            self.coherence.last_time = current_time
            self.coherence.tick_count += 1
            
            # Print status every second
            if self.coherence.tick_count % 30 == 0:
                print(f"Tick {self.coherence.tick_count} | "
                      f"Reality: {self.coherence.reality_field:.3f} | "
                      f"Trust: C={self.coherence.trust_weights['camera']:.2f} "
                      f"I={self.coherence.trust_weights['imu']:.2f}")
                      
        # Finalize
        self.finalize_experiment()
        
    def finalize_experiment(self):
        """Clean up and analyze results"""
        print("\n" + "="*60)
        print("FINALIZING EXPERIMENT")
        print("="*60)
        
        # Finalize logging
        stats = self.logger.finalize()
        print(f"\nLogged {stats['entries_logged']} data points")
        
        # Analyze results
        print("\nAnalyzing trust dynamics...")
        analysis = self.logger.analyze()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        print(f"\nTotal Duration: {analysis.get('duration', 0):.1f} seconds")
        print(f"Total Entries: {analysis.get('total_entries', 0)}")
        print(f"Conflicts Detected: {analysis.get('conflicts', 0)}")
        
        print("\nTrust Weight Changes:")
        for sensor, changes in analysis.get('trust_changes', {}).items():
            print(f"  {sensor}:")
            print(f"    Initial: {changes['initial']:.3f}")
            print(f"    Final: {changes['final']:.3f}")
            print(f"    Change: {changes['change']:+.3f}")
            
        print("\nPhase Analysis:")
        for phase, data in analysis.get('phases', {}).items():
            print(f"  {phase}:")
            print(f"    Entries: {data['count']}")
            print(f"    Avg Reality Field: {data['avg_reality_field']:.3f}")
            
        print("\nContext Transitions:")
        for transition in analysis.get('context_transitions', [])[:5]:
            print(f"  {transition['from']} → {transition['to']} at {transition['time']:.1f}s")
            
        # Shutdown coherence engine
        self.coherence.shutdown()
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[INTERRUPT] Stopping experiment...")
        self.experiment_running = False
        self.coherence.running = False


def main():
    experiment = TrustDynamicsExperiment()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, experiment.signal_handler)
    
    try:
        experiment.run_experiment()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(experiment, 'coherence'):
            experiment.coherence.shutdown()


if __name__ == "__main__":
    main()