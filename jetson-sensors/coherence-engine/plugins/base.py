"""
Base classes for Coherence Engine plugins
Implements MCP-like architecture with sensor-effector duality
August 11, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import hashlib
from datetime import datetime

class PluginBase(ABC):
    """Base class for all plugins with LCT integration"""
    
    def __init__(self, identity: str):
        self.identity = identity
        self.lct = self._generate_lct(identity)
        self.created_at = datetime.now()
        self.trust_weight = 1.0
        self.relevance_weight = 1.0
        
    def _generate_lct(self, identity: str) -> Dict[str, Any]:
        """Generate Linked Context Token for this plugin"""
        return {
            "id": hashlib.sha256(identity.encode()).hexdigest()[:16],
            "type": self.__class__.__name__,
            "identity": identity,
            "created": datetime.now().isoformat(),
            "t3": {  # Talent/Training/Temperament
                "talent": 1.0,
                "training": 0.0,
                "temperament": 0.5
            },
            "v3": {  # Value/Verification/Validity
                "value": 1.0,
                "verification": 1.0,
                "validity": 1.0
            },
            "mrh": "device"  # Markov Relevancy Horizon
        }
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    def teardown(self):
        """Clean shutdown of plugin"""
        pass
    
    def get_lct(self) -> Dict[str, Any]:
        """Return the plugin's LCT"""
        return self.lct
    
    def get_identity(self) -> str:
        """Return the plugin's identity string"""
        return self.identity
    
    def update_trust(self, delta: float):
        """Update trust weight based on performance"""
        self.trust_weight = max(0.1, min(1.0, self.trust_weight + delta))
        self.lct["t3"]["training"] += abs(delta) * 0.1  # Learning from experience
    
    def update_relevance(self, context: str, weight: float):
        """Update relevance for specific context"""
        self.relevance_weight = weight
        

class SensorBase(PluginBase):
    """Base class for sensor plugins"""
    
    @abstractmethod
    def read(self) -> Any:
        """Read sensor data - main sensing interface"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Declare sensor capabilities"""
        pass
    
    def process(self, input_data: Optional[Any] = None) -> Any:
        """Process method for compatibility with MCP pattern"""
        return self.read()
    

class EffectorBase(PluginBase):
    """Base class for effector plugins"""
    
    @abstractmethod
    def execute(self, action: Dict[str, Any]) -> bool:
        """Execute an action - main effector interface"""
        pass
    
    @abstractmethod
    def propose_action(self, reality_field: Any, goal_state: Any) -> Dict[str, Any]:
        """Propose an action based on current reality and goal"""
        pass
    
    @abstractmethod
    def get_energy_cost(self) -> float:
        """Report energy cost of actions"""
        pass
    
    @abstractmethod
    def predict_outcome(self, action: Dict[str, Any]) -> Any:
        """Predict the outcome of an action"""
        pass
    

class SensorEffectorBridge(SensorBase, EffectorBase):
    """
    Implements sensor-effector duality
    Every sensor output can be an effector at its MRH level
    """
    
    def __init__(self, identity: str):
        super().__init__(identity)
        self.last_reading = None
        
    def execute(self, action: Dict[str, Any]) -> bool:
        """Default: sensor output becomes effector action"""
        # Example: A memory sensor's output can write to memory
        # A vision sensor's output can be displayed
        return True
    
    def get_energy_cost(self) -> float:
        """Energy cost of using sensor as effector"""
        return 0.01  # Minimal by default
    
    def predict_outcome(self, action: Dict[str, Any]) -> Any:
        """Predict outcome when sensor acts as effector"""
        return self.last_reading