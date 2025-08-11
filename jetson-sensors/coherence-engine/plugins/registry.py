"""
Plugin Registry with manifest-based discovery
Lightweight for embedded systems (no pkg_resources)
"""
from importlib import import_module
import json
import pathlib
from typing import Dict, Any, Type
import logging

logger = logging.getLogger(__name__)

class Registry:
    def __init__(self, root: str = "plugins"):
        self.root = pathlib.Path(root)
        self.registry: Dict[str, Dict[str, Any]] = {}  # lct -> {"class": cls, "manifest": manifest}
        
    def discover(self) -> Dict[str, Dict[str, Any]]:
        """Scan for plugin manifests and dynamically load classes"""
        discovered = {}
        
        # Find all plugin.json files
        for manifest_path in self.root.glob("*/plugin.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                lct = manifest["lct"]
                
                # Dynamic import
                module = import_module(manifest["module"])
                plugin_class = getattr(module, manifest["class"])
                
                # Store both class and manifest
                self.registry[lct] = {
                    "class": plugin_class,
                    "manifest": manifest,
                    "path": manifest_path.parent
                }
                
                discovered[lct] = manifest
                logger.info(f"Discovered plugin: {lct} from {manifest_path}")
                
            except Exception as e:
                logger.error(f"Failed to load plugin from {manifest_path}: {e}")
                continue
                
        return discovered
    
    def get_class(self, lct: str) -> Type:
        """Get plugin class by LCT"""
        if lct not in self.registry:
            raise KeyError(f"Plugin {lct} not found in registry")
        return self.registry[lct]["class"]
    
    def get_manifest(self, lct: str) -> Dict[str, Any]:
        """Get plugin manifest by LCT"""
        if lct not in self.registry:
            raise KeyError(f"Plugin {lct} not found in registry")
        return self.registry[lct]["manifest"]
    
    def list_plugins(self) -> list:
        """List all discovered plugin LCTs"""
        return list(self.registry.keys())