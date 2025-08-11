"""
Display effector plugin - corrected with instance handling
"""
import time
from typing import Dict, Any, List
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from plugins.base_v2 import EffectorBase

logger = logging.getLogger(__name__)

class DisplayEffector(EffectorBase):
    """HDMI display output effector for Jetson"""
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize with manifest"""
        super().__init__(manifest)
        self.refresh_rate = None
        self.resolution = None
        self.hdmi_output = None
        self.overlay_alpha = None
        self.frame_count = 0
        self.last_update_time = time.time()
        self.current_overlays = []
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize the display effector"""
        super().initialize(config)
        
        # Extract config
        self.refresh_rate = config.get("refresh_rate", 60)
        self.resolution = tuple(config.get("resolution", [1920, 1080]))
        self.hdmi_output = config.get("hdmi_output", True)
        self.overlay_alpha = config.get("overlay_alpha", 0.5)
        
        logger.info(f"Initialized DisplayEffector {self.lct}: {self.resolution}@{self.refresh_rate}Hz")
        
        # In real implementation, initialize display output here
        # self.display = create_hdmi_output(self.resolution, self.refresh_rate)
        
    def execute(self, action: Dict[str, Any]) -> bool:
        """Execute display action"""
        action_type = action.get("type", "overlay")
        
        try:
            if action_type == "overlay":
                return self._draw_overlay(action)
            elif action_type == "clear":
                return self._clear_display()
            elif action_type == "attention_box":
                return self._draw_attention_box(action)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False
    
    def _draw_overlay(self, action: Dict[str, Any]) -> bool:
        """Draw overlay on display"""
        overlay_data = action.get("data", {})
        
        # Enforce refresh rate
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        min_interval = 1.0 / self.refresh_rate
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        # Add to overlay list
        self.current_overlays.append({
            "type": "overlay",
            "data": overlay_data,
            "timestamp": current_time
        })
        
        # Keep only recent overlays
        if len(self.current_overlays) > 10:
            self.current_overlays = self.current_overlays[-10:]
        
        self.last_update_time = time.time()
        self.frame_count += 1
        
        # In real impl: draw to display
        # self.display.draw_overlay(overlay_data, self.overlay_alpha)
        
        return True
    
    def _draw_attention_box(self, action: Dict[str, Any]) -> bool:
        """Draw attention box on display"""
        box = action.get("box", {})
        x = box.get("x", 0)
        y = box.get("y", 0)
        w = box.get("width", 100)
        h = box.get("height", 100)
        color = box.get("color", [255, 0, 0])  # Red default
        
        self.current_overlays.append({
            "type": "attention_box",
            "box": {"x": x, "y": y, "w": w, "h": h},
            "color": color,
            "timestamp": time.time()
        })
        
        logger.debug(f"Drew attention box at ({x},{y}) size ({w},{h})")
        
        # In real impl: draw box on display
        # self.display.draw_rectangle(x, y, w, h, color)
        
        return True
    
    def _clear_display(self) -> bool:
        """Clear all overlays"""
        self.current_overlays = []
        logger.debug("Cleared display overlays")
        
        # In real impl: clear display
        # self.display.clear()
        
        return True
    
    def propose_action(self, reality_field: Dict[str, Any], goal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Propose display action based on reality and goals"""
        # Example: If motion detected, propose attention box
        motion = reality_field.get("motion", {})
        if motion.get("detected"):
            regions = motion.get("regions", [])
            if regions:
                region = regions[0]
                return {
                    "type": "attention_box",
                    "box": {
                        "x": region.get("x", 0),
                        "y": region.get("y", 0),
                        "width": region.get("w", 50),
                        "height": region.get("h", 50)
                    },
                    "color": [255, 255, 0],  # Yellow for motion
                    "priority": 0.8
                }
        
        # Default: no action needed
        return {"type": "none", "priority": 0}
    
    def predict_outcome(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome of display action"""
        action_type = action.get("type")
        
        if action_type == "attention_box":
            return {
                "visual_change": True,
                "user_attention_likely": True,
                "energy_cost_mw": 10
            }
        elif action_type == "overlay":
            return {
                "visual_change": True,
                "user_attention_likely": False,
                "energy_cost_mw": 5
            }
        else:
            return {
                "visual_change": False,
                "user_attention_likely": False,
                "energy_cost_mw": 0
            }
    
    def get_display_state(self) -> Dict[str, Any]:
        """Get current display state"""
        return {
            "frame_count": self.frame_count,
            "overlays": len(self.current_overlays),
            "last_update": self.last_update_time,
            "fps": self.frame_count / (time.time() - self.last_update_time) if self.frame_count > 0 else 0
        }
    
    def teardown(self):
        """Clean up display resources"""
        logger.info(f"Shutting down DisplayEffector {self.lct}")
        self._clear_display()
        # In real impl: release display
        # self.display.release()