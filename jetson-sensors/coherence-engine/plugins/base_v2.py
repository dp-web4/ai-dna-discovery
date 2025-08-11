"""
Corrected base classes for plugins with proper instance handling
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import hashlib
import json
import numpy as np

class TransportBase(ABC):
    """Base class for transport mechanisms"""
    @abstractmethod
    def send(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def receive(self) -> Any:
        pass

class InProcTransport(TransportBase):
    """Direct in-process call (zero-copy)"""
    def send(self, data: Any) -> Any:
        return data  # Direct pass-through
    
    def receive(self) -> Any:
        return None  # Not used in direct mode

class PluginBase(ABC):
    """Base class for all plugins with manifest support"""
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize with manifest data"""
        self.manifest = manifest
        self.lct = manifest["lct"]
        self.capabilities = manifest.get("capabilities", {})
        self.config = {}
        self.transport = self._setup_transport(manifest.get("transport", "inproc"))
        
        # Trust and relevance weights (can be updated at runtime)
        self.trust_weight = 1.0
        self.relevance_weight = 1.0
        
    def _setup_transport(self, transport_type: str) -> TransportBase:
        """Setup transport based on manifest"""
        if transport_type == "inproc":
            return InProcTransport()
        # Add other transports (UDS, MsgPack) as needed
        else:
            return InProcTransport()  # Default
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin with configuration"""
        self.config = config
    
    @abstractmethod
    def teardown(self):
        """Clean up resources"""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities"""
        return self.capabilities
    
    def get_lct(self) -> str:
        """Return instance LCT"""
        return self.lct

class SensorBase(PluginBase):
    """Base class for sensor plugins"""
    
    @abstractmethod
    def read(self) -> Any:
        """Read sensor data"""
        pass
    
    def process(self, input_data: Optional[Any] = None) -> Any:
        """Process input and return sensor data"""
        return self.read()

class EffectorBase(PluginBase):
    """Base class for effector plugins"""
    
    @abstractmethod
    def execute(self, action: Dict[str, Any]) -> bool:
        """Execute an action"""
        pass
    
    @abstractmethod
    def propose_action(self, reality_field: Dict[str, Any], goal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Propose an action based on reality and goals"""
        pass
    
    def get_energy_cost(self) -> float:
        """Return estimated energy cost in milliwatts"""
        return self.capabilities.get("energy_hint_mw", 0)
    
    def predict_outcome(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the outcome of an action"""
        return {"predicted": "unknown"}

class SensorEffectorBridge(SensorBase, EffectorBase):
    """Base for components that are both sensors and effectors"""
    
    def __init__(self, manifest: Dict[str, Any]):
        super().__init__(manifest)
        self.last_output = None
    
    def read(self) -> Any:
        """Read current state"""
        return self.last_output
    
    def execute(self, action: Dict[str, Any]) -> bool:
        """Execute affects what we sense"""
        self.last_output = action.get("data")
        return True