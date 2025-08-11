"""
Plugin Manager for Coherence Engine
Handles discovery, lifecycle, and communication
August 11, 2025
"""

import os
import sys
import json
import importlib
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import queue

from base import PluginBase, SensorBase, EffectorBase

class PluginManager:
    """Manages plugin lifecycle and communication"""
    
    def __init__(self, coherence_engine=None):
        self.engine = coherence_engine
        self.registered_plugins = {}  # LCT -> plugin class
        self.active_plugins = {}  # LCT -> plugin instance
        self.plugin_paths = []
        self.config = {}
        self.message_queue = queue.Queue()
        self.running = True
        
        # Start message handler thread
        self.handler_thread = threading.Thread(target=self._message_handler)
        self.handler_thread.daemon = True
        self.handler_thread.start()
    
    def add_plugin_path(self, path: str):
        """Add a path to search for plugins"""
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
            if path not in sys.path:
                sys.path.append(path)
    
    def discover_plugins(self):
        """Discover all available plugins in plugin paths"""
        discovered = []
        
        for plugin_path in self.plugin_paths:
            path = Path(plugin_path)
            if not path.exists():
                continue
                
            for py_file in path.glob("*.py"):
                if py_file.stem.startswith("_"):
                    continue
                    
                try:
                    # Import the module
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, PluginBase) and 
                            attr not in [PluginBase, SensorBase, EffectorBase]):
                            discovered.append(attr)
                            print(f"Discovered plugin: {attr.__name__}")
                            
                except Exception as e:
                    print(f"Error loading plugin from {py_file}: {e}")
        
        return discovered
    
    def register_plugin(self, plugin_class):
        """Register a plugin class"""
        try:
            # Create temporary instance to get LCT
            temp_instance = plugin_class(f"temp_{plugin_class.__name__}")
            lct_id = temp_instance.get_lct()["id"]
            
            self.registered_plugins[lct_id] = plugin_class
            print(f"Registered plugin: {plugin_class.__name__} with LCT: {lct_id}")
            
            return lct_id
        except Exception as e:
            print(f"Error registering plugin {plugin_class.__name__}: {e}")
            return None
    
    def start_plugin(self, lct_id: str, config: Optional[Dict] = None):
        """Start a plugin instance"""
        if lct_id not in self.registered_plugins:
            raise ValueError(f"Plugin {lct_id} not registered")
        
        if lct_id in self.active_plugins:
            print(f"Plugin {lct_id} already active")
            return
        
        try:
            plugin_class = self.registered_plugins[lct_id]
            plugin_instance = plugin_class(lct_id)
            
            # Initialize with config
            if config:
                plugin_instance.initialize(config)
            else:
                plugin_instance.initialize({})
            
            self.active_plugins[lct_id] = plugin_instance
            print(f"Started plugin: {lct_id}")
            
            # Notify coherence engine if available
            if self.engine:
                if isinstance(plugin_instance, SensorBase):
                    self.engine.add_sensor(plugin_instance)
                if isinstance(plugin_instance, EffectorBase):
                    self.engine.add_effector(plugin_instance)
                    
        except Exception as e:
            print(f"Error starting plugin {lct_id}: {e}")
            traceback.print_exc()
    
    def stop_plugin(self, lct_id: str):
        """Stop a plugin instance"""
        if lct_id not in self.active_plugins:
            print(f"Plugin {lct_id} not active")
            return
        
        try:
            plugin_instance = self.active_plugins[lct_id]
            plugin_instance.teardown()
            del self.active_plugins[lct_id]
            print(f"Stopped plugin: {lct_id}")
            
            # Notify coherence engine if available
            if self.engine:
                if isinstance(plugin_instance, SensorBase):
                    self.engine.remove_sensor(plugin_instance)
                if isinstance(plugin_instance, EffectorBase):
                    self.engine.remove_effector(plugin_instance)
                    
        except Exception as e:
            print(f"Error stopping plugin {lct_id}: {e}")
    
    def communicate(self, lct_id: str, method: str, *args, **kwargs):
        """Send a message to a plugin"""
        if lct_id not in self.active_plugins:
            raise ValueError(f"Plugin {lct_id} not active")
        
        try:
            plugin = self.active_plugins[lct_id]
            method_func = getattr(plugin, method)
            return method_func(*args, **kwargs)
        except Exception as e:
            print(f"Error communicating with plugin {lct_id}: {e}")
            self.handle_plugin_failure(lct_id, e)
            return None
    
    def broadcast(self, method: str, *args, **kwargs):
        """Broadcast a message to all active plugins"""
        results = {}
        for lct_id in self.active_plugins:
            try:
                result = self.communicate(lct_id, method, *args, **kwargs)
                results[lct_id] = result
            except:
                pass
        return results
    
    def handle_plugin_failure(self, lct_id: str, error: Exception):
        """Handle plugin failure"""
        print(f"Plugin {lct_id} failed: {error}")
        
        # Update trust weight
        if lct_id in self.active_plugins:
            plugin = self.active_plugins[lct_id]
            plugin.update_trust(-0.1)
        
        # Restart plugin if critical
        if self.is_critical_plugin(lct_id):
            print(f"Attempting to restart critical plugin {lct_id}")
            self.stop_plugin(lct_id)
            self.start_plugin(lct_id, self.config.get(lct_id, {}))
    
    def is_critical_plugin(self, lct_id: str) -> bool:
        """Determine if a plugin is critical"""
        # Can be configured per plugin
        return False
    
    def _message_handler(self):
        """Background thread for handling async messages"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.1)
                if message:
                    self._process_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in message handler: {e}")
    
    def _process_message(self, message: Dict[str, Any]):
        """Process an async message"""
        # Implement async message processing
        pass
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Add plugin paths
        for path in self.config.get("plugin_discovery_paths", []):
            self.add_plugin_path(path)
        
        # Configure plugins
        for lct_id, plugin_config in self.config.get("plugins", {}).items():
            if lct_id in self.active_plugins:
                # Update running plugin
                plugin = self.active_plugins[lct_id]
                for key, value in plugin_config.items():
                    setattr(plugin, key, value)
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins"""
        status = {
            "registered": list(self.registered_plugins.keys()),
            "active": list(self.active_plugins.keys()),
            "plugin_paths": self.plugin_paths,
            "details": {}
        }
        
        for lct_id, plugin in self.active_plugins.items():
            status["details"][lct_id] = {
                "type": plugin.__class__.__name__,
                "trust_weight": plugin.trust_weight,
                "relevance_weight": plugin.relevance_weight,
                "lct": plugin.get_lct()
            }
        
        return status
    
    def shutdown(self):
        """Shutdown the plugin manager"""
        self.running = False
        
        # Stop all plugins
        for lct_id in list(self.active_plugins.keys()):
            self.stop_plugin(lct_id)
        
        # Wait for handler thread
        if self.handler_thread.is_alive():
            self.handler_thread.join(timeout=1.0)