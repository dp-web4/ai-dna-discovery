"""
Visual Dashboard Effector Plugin for Coherence Engine
Real-time visualization of reality field and sensor data
August 12, 2025
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List
import threading
import queue
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.base import EffectorBase

class DashboardEffectorPlugin(EffectorBase):
    """Visual dashboard for real-time coherence monitoring"""
    
    def __init__(self, identity: str = "dashboard_effector"):
        super().__init__(identity)
        self.window_name = "Coherence Engine Dashboard"
        self.display_size = (1920, 1080)
        self.running = False
        self.display_thread = None
        self.update_queue = queue.Queue(maxsize=100)  # Larger queue for smoother updates
        
        # Dashboard state
        self.reality_field = 0.0
        self.sensor_data = {}
        self.context_state = "STABLE"
        self.trust_weights = {}
        self.relevance_weights = {}
        self.fps_history = deque(maxlen=30)
        self.attention_regions = []
        
        # Visual elements
        self.camera_frames = [None, None]
        self.last_update = time.time()
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize OpenCV window and display thread"""
        self.display_size = config.get("display_size", self.display_size)
        
        print(f"Initializing dashboard effector: {self.display_size[0]}x{self.display_size[1]}")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_size[0], self.display_size[1])
        
        # Start display thread
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def teardown(self):
        """Clean up display resources"""
        self.running = False
        
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
            
        cv2.destroyWindow(self.window_name)
        print("Dashboard effector shutdown complete")
        
    def _display_loop(self):
        """Main display loop"""
        while self.running:
            # Process update queue first to get latest data
            while not self.update_queue.empty():
                try:
                    update = self.update_queue.get_nowait()
                    self._apply_update(update)
                except queue.Empty:
                    break
            
            # Create dashboard frame with current data
            dashboard = self._create_dashboard()
                    
            # Display dashboard
            cv2.imshow(self.window_name, dashboard)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_{timestamp}.png"
                cv2.imwrite(filename, dashboard)
                print(f"Screenshot saved: {filename}")
                
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_update + 0.001)
            self.fps_history.append(fps)
            self.last_update = current_time
            
            # Maintain ~30 FPS
            time.sleep(0.033)
            
    def _create_dashboard(self) -> np.ndarray:
        """Create the dashboard visualization"""
        # Create base canvas
        dashboard = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark gray background
        
        # Layout: 
        # Top: Camera feeds (if available)
        # Middle: Reality field visualization
        # Bottom: Sensor data and metrics
        
        # Draw camera feeds (top half)
        if any(self.camera_frames):
            self._draw_camera_feeds(dashboard)
            
        # Draw reality field (middle)
        self._draw_reality_field(dashboard)
        
        # Draw sensor metrics (bottom)
        self._draw_sensor_metrics(dashboard)
        
        # Draw attention regions
        self._draw_attention_regions(dashboard)
        
        # Draw FPS counter
        self._draw_fps(dashboard)
        
        return dashboard
        
    def _draw_camera_feeds(self, dashboard: np.ndarray):
        """Draw camera feeds on dashboard"""
        y_offset = 20
        
        for i, frame in enumerate(self.camera_frames):
            if frame is not None and len(frame.shape) == 3:  # Valid frame
                # Ensure frame is correct size (960x540)
                if frame.shape[:2] != (540, 960):
                    frame = cv2.resize(frame, (960, 540))
                
                # Position for each camera
                if i == 0:  # Left camera
                    x_offset = 20
                else:  # Right camera  
                    x_offset = 940
                
                # Place frame on dashboard (ensure it fits)
                h, w = frame.shape[:2]
                end_y = min(y_offset + h, dashboard.shape[0])
                end_x = min(x_offset + w, dashboard.shape[1])
                
                # Copy frame to dashboard
                dashboard[y_offset:end_y, x_offset:end_x] = frame[:end_y-y_offset, :end_x-x_offset]
                
                # Add label
                label = f"Camera {i}" if i < 2 else f"Camera {i}"
                cv2.putText(dashboard, label, (x_offset + 10, y_offset + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
    def _draw_reality_field(self, dashboard: np.ndarray):
        """Draw reality field visualization"""
        # Position in middle of screen
        center_y = self.display_size[1] // 2
        center_x = self.display_size[0] // 2
        
        # Draw reality field as circular visualization
        radius = int(100 * (1 + self.reality_field))
        color_intensity = int(255 * min(self.reality_field, 1.0))
        color = (0, color_intensity, 255 - color_intensity)
        
        cv2.circle(dashboard, (center_x, center_y), radius, color, -1)
        cv2.circle(dashboard, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Add coherence text
        text = f"Reality Field: {self.reality_field:.3f}"
        cv2.putText(dashboard, text, (center_x - 100, center_y - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        # Add context state
        context_color = {
            "STABLE": (0, 255, 0),
            "MOVING": (255, 255, 0),
            "UNSTABLE": (255, 165, 0),
            "NOVEL": (255, 0, 255)
        }.get(self.context_state, (255, 255, 255))
        
        cv2.putText(dashboard, f"Context: {self.context_state}", 
                   (center_x - 80, center_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, context_color, 2)
                   
    def _draw_sensor_metrics(self, dashboard: np.ndarray):
        """Draw sensor data and metrics"""
        # Position at bottom of screen
        y_start = self.display_size[1] - 300
        x_start = 20
        
        # Title
        cv2.putText(dashboard, "Sensor Contributions", (x_start, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                   
        # Draw each sensor's contribution
        y_offset = y_start + 40
        for sensor_name, data in self.sensor_data.items():
            # Get weights
            trust = self.trust_weights.get(sensor_name, 1.0)
            relevance = self.relevance_weights.get(sensor_name, 1.0)
            contribution = data.get("value", 0.0) * trust * relevance
            
            # Draw bar graph
            bar_width = int(200 * abs(contribution))
            bar_color = (0, 255, 0) if contribution >= 0 else (0, 0, 255)
            
            cv2.rectangle(dashboard, (x_start + 150, y_offset - 15),
                         (x_start + 150 + bar_width, y_offset + 5),
                         bar_color, -1)
                         
            # Draw text
            text = f"{sensor_name}: {contribution:.3f} (T:{trust:.2f} R:{relevance:.2f})"
            cv2.putText(dashboard, text, (x_start, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                       
            y_offset += 30
            
    def _draw_attention_regions(self, dashboard: np.ndarray):
        """Draw attention boxes on dashboard"""
        for region in self.attention_regions:
            x, y, w, h = region["coords"]
            color = region.get("color", (255, 0, 0))
            thickness = region.get("thickness", 2)
            
            cv2.rectangle(dashboard, (x, y), (x + w, y + h), color, thickness)
            
            # Add label if provided
            if "label" in region:
                cv2.putText(dashboard, region["label"], (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                           
    def _draw_fps(self, dashboard: np.ndarray):
        """Draw FPS counter"""
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            text = f"FPS: {avg_fps:.1f}"
            cv2.putText(dashboard, text, (self.display_size[0] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                       
    def _apply_update(self, update: Dict[str, Any]):
        """Apply an update to the dashboard state"""
        update_type = update.get("type")
        
        if update_type == "reality_field":
            self.reality_field = update.get("value", 0.0)
            self.context_state = update.get("context", "STABLE")
            
        elif update_type == "sensor_data":
            self.sensor_data = update.get("sensors", {})
            self.trust_weights = update.get("trust", {})
            self.relevance_weights = update.get("relevance", {})
            
        elif update_type == "camera_frames":
            self.camera_frames = update.get("frames", [None, None])
            
        elif update_type == "attention":
            self.attention_regions = update.get("regions", [])
            
    def execute(self, action: Dict[str, Any]) -> bool:
        """Execute dashboard update action"""
        action_type = action.get("type")
        
        if action_type == "update":
            # Queue update for display thread
            try:
                self.update_queue.put_nowait(action)
                return True
            except queue.Full:
                # Drop update if queue is full
                return False
                
        elif action_type == "alert":
            # Show alert on dashboard
            alert_update = {
                "type": "attention",
                "regions": [{
                    "coords": [self.display_size[0]//2 - 200, 100, 400, 100],
                    "color": (0, 0, 255),
                    "thickness": 3,
                    "label": action.get("message", "ALERT")
                }]
            }
            self.update_queue.put(alert_update)
            return True
            
        elif action_type == "screenshot":
            # Trigger screenshot on next frame
            # Handled in display loop with 's' key
            return True
            
        return False
        
    def propose_action(self, reality_field: Any, goal_state: Any) -> Dict[str, Any]:
        """Propose dashboard update based on reality field"""
        # Always update with current reality field
        return {
            "type": "update",
            "update_type": "reality_field",
            "value": reality_field.get("coherence", 0.0),
            "context": reality_field.get("context", "STABLE")
        }
        
    def get_energy_cost(self) -> float:
        """Report energy cost of dashboard operation"""
        # Base display cost
        base_cost = 0.01
        
        # Additional cost for camera display
        camera_cost = sum(0.005 for frame in self.camera_frames if frame is not None)
        
        return base_cost + camera_cost
        
    def predict_outcome(self, action: Dict[str, Any]) -> Any:
        """Predict outcome of dashboard action"""
        if action.get("type") == "update":
            return {"display_updated": True, "latency": 0.033}  # ~30 FPS
            
        elif action.get("type") == "alert":
            return {"user_notified": True, "attention_captured": 0.9}
            
        return {"action_executed": True}
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Declare dashboard capabilities"""
        return {
            "type": "display",
            "subtype": "visual_dashboard",
            "resolution": self.display_size,
            "features": [
                "reality_field_visualization",
                "sensor_metrics",
                "camera_feeds",
                "attention_regions",
                "fps_counter"
            ],
            "update_rate": 30,  # FPS
            "interaction": ["screenshot", "quit"]
        }