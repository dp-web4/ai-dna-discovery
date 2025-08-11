"""
Enhanced Plugin Manager with proper instance management and lifecycle
"""
import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional
from collections import deque
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

class PluginState(Enum):
    """Plugin lifecycle states"""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    QUARANTINED = "quarantined"

@dataclass
class PluginMetrics:
    """Runtime metrics for a plugin"""
    call_count: int = 0
    error_count: int = 0
    last_call_time: float = 0
    total_latency: float = 0
    state_transitions: list = field(default_factory=list)
    
    def record_call(self, latency: float):
        self.call_count += 1
        self.total_latency += latency
        self.last_call_time = time.time()
    
    def record_error(self):
        self.error_count += 1
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.call_count if self.call_count > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count > 0 else 0

class PluginManager:
    def __init__(self, registry, max_errors: int = 5):
        self.registry = registry
        self.running: Dict[str, Any] = {}  # lct -> instance
        self.states: Dict[str, PluginState] = {}  # lct -> state
        self.metrics: Dict[str, PluginMetrics] = {}  # lct -> metrics
        self.queues: Dict[str, asyncio.Queue] = {}  # lct -> queue for async
        self.max_errors = max_errors
        
    def _transition_state(self, lct: str, new_state: PluginState):
        """Record state transition"""
        old_state = self.states.get(lct, PluginState.DISCOVERED)
        self.states[lct] = new_state
        
        if lct not in self.metrics:
            self.metrics[lct] = PluginMetrics()
            
        transition = {
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics[lct].state_transitions.append(transition)
        logger.info(f"Plugin {lct}: {old_state.value} -> {new_state.value}")
    
    def start(self, lct: str, runtime_config: Optional[Dict] = None):
        """Start a plugin instance"""
        if lct in self.running:
            logger.warning(f"Plugin {lct} already running")
            return
            
        try:
            # Get class and manifest from registry
            meta = self.registry.registry[lct]
            plugin_class = meta["class"]
            manifest = meta["manifest"]
            
            # Create instance
            instance = plugin_class(manifest=manifest)
            self._transition_state(lct, PluginState.LOADED)
            
            # Initialize with config
            config = manifest.get("config", {})
            if runtime_config:
                config.update(runtime_config)
            instance.initialize(config)
            self._transition_state(lct, PluginState.INITIALIZED)
            
            # Store instance
            self.running[lct] = instance
            
            # Create async queue if needed
            capabilities = manifest.get("capabilities", {})
            if capabilities.get("async", False):
                queue_size = capabilities.get("queue_size", 100)
                self.queues[lct] = asyncio.Queue(maxsize=queue_size)
            
            self._transition_state(lct, PluginState.RUNNING)
            logger.info(f"Started plugin: {lct}")
            
        except Exception as e:
            logger.error(f"Failed to start plugin {lct}: {e}")
            self._transition_state(lct, PluginState.STOPPED)
            raise
    
    def stop(self, lct: str):
        """Stop a plugin instance"""
        if lct not in self.running:
            logger.warning(f"Plugin {lct} not running")
            return
            
        try:
            instance = self.running[lct]
            instance.teardown()
            del self.running[lct]
            
            if lct in self.queues:
                del self.queues[lct]
                
            self._transition_state(lct, PluginState.STOPPED)
            logger.info(f"Stopped plugin: {lct}")
            
        except Exception as e:
            logger.error(f"Error stopping plugin {lct}: {e}")
            self._transition_state(lct, PluginState.QUARANTINED)
    
    def call(self, lct: str, method: str, *args, **kwargs):
        """Call a method on a running plugin instance with metrics"""
        if lct not in self.running:
            raise RuntimeError(f"Plugin {lct} not running")
            
        instance = self.running[lct]
        
        # Check if degraded or quarantined
        state = self.states.get(lct, PluginState.STOPPED)
        if state == PluginState.QUARANTINED:
            raise RuntimeError(f"Plugin {lct} is quarantined")
        
        # Measure latency
        start_time = time.time()
        
        try:
            # Get method and call it
            method_func = getattr(instance, method)
            result = method_func(*args, **kwargs)
            
            # Record metrics
            latency = time.time() - start_time
            self.metrics[lct].record_call(latency)
            
            # Check latency budget
            manifest = self.registry.get_manifest(lct)
            budget_ms = manifest.get("capabilities", {}).get("latency_budget_ms", 1000)
            if latency * 1000 > budget_ms:
                logger.warning(f"Plugin {lct}.{method} exceeded latency budget: {latency*1000:.2f}ms > {budget_ms}ms")
            
            # Reset to running if was degraded and call succeeded
            if state == PluginState.DEGRADED:
                self._transition_state(lct, PluginState.RUNNING)
                
            return result
            
        except Exception as e:
            # Record error
            self.metrics[lct].record_error()
            
            # Check error threshold
            if self.metrics[lct].error_count >= self.max_errors:
                logger.error(f"Plugin {lct} exceeded error threshold, quarantining")
                self._transition_state(lct, PluginState.QUARANTINED)
            elif self.metrics[lct].error_rate > 0.3:  # >30% error rate
                logger.warning(f"Plugin {lct} degraded due to high error rate")
                self._transition_state(lct, PluginState.DEGRADED)
                
            raise
    
    async def call_async(self, lct: str, method: str, *args, **kwargs):
        """Async call with backpressure handling"""
        if lct not in self.queues:
            # Fallback to sync call
            return self.call(lct, method, *args, **kwargs)
            
        queue = self.queues[lct]
        
        # Check queue pressure
        if queue.full():
            logger.warning(f"Plugin {lct} queue full, applying backpressure")
            # Could implement drop policy here
            
        # Queue the call
        await queue.put((method, args, kwargs))
        
        # Process queue (in real impl, this would be a worker task)
        method, args, kwargs = await queue.get()
        return self.call(lct, method, *args, **kwargs)
    
    def get_metrics(self, lct: str) -> PluginMetrics:
        """Get runtime metrics for a plugin"""
        return self.metrics.get(lct, PluginMetrics())
    
    def get_state(self, lct: str) -> PluginState:
        """Get current state of a plugin"""
        return self.states.get(lct, PluginState.DISCOVERED)
    
    def list_running(self) -> list:
        """List all running plugin LCTs"""
        return list(self.running.keys())