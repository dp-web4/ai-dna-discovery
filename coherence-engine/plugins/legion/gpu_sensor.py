"""
GPU sensor for Legion (RTX 4090) - monitors GPU utilization and temperature
"""
import subprocess
import re
import os
from typing import Optional

# Try to import nvidia-ml-py for fallback
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class GPUSensor:
    """GPU utilization and temperature sensor using nvidia-smi"""
    
    def __init__(self):
        self.id = "gpu"
        self.available = False
        self.use_nvml = False
        self.device_handle = None
        self.last_util = 0.0
        self.last_temp = 0.0
        
        # Try nvidia-smi first
        if self.check_nvidia_smi():
            self.available = True
        # Try NVML as fallback
        elif NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.use_nvml = True
                self.available = True
                name = pynvml.nvmlDeviceGetName(self.device_handle)
                print(f"    ✓ Found: {name.decode() if isinstance(name, bytes) else name} (via NVML)")
            except Exception as e:
                print(f"    NVML init failed: {e}")
        # Last resort - check for device files
        elif os.path.exists('/dev/nvidia0'):
            self.available = True
            print("    ✓ Found: NVIDIA GPU (device files only - limited functionality)")
        
    def check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--help'], 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def read(self, *, tick: int) -> float:
        """
        Read GPU sensor data.
        Returns normalized [0,1] value based on GPU utilization and temperature.
        """
        if not self.available:
            return 0.0
        
        # Try NVML first if available
        if self.use_nvml and self.device_handle:
            try:
                # Get utilization
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle)
                util = util_rates.gpu / 100.0  # Convert to [0,1]
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.device_handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_norm = max(0.0, min(1.0, (temp - 30) / 55))  # Normalize 30-85°C
                
                self.last_util = util
                self.last_temp = temp_norm
                
                # Return weighted combination
                return util * 0.7 + temp_norm * 0.3
                
            except Exception as e:
                # Fall through to nvidia-smi
                pass
        
        # Try nvidia-smi
        if not self.use_nvml:
            try:
                # Query GPU utilization and temperature
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        parts = output.split(',')
                        if len(parts) >= 2:
                            util = float(parts[0].strip()) / 100.0  # Convert percentage to [0,1]
                            temp = float(parts[1].strip())
                            
                            # Normalize temperature (assume 30-85°C range)
                            temp_norm = max(0.0, min(1.0, (temp - 30) / 55))
                            
                            # Combine utilization and temperature
                            # High utilization is activity, high temp might indicate stress
                            self.last_util = util
                            self.last_temp = temp_norm
                            
                            # Return weighted combination
                            return util * 0.7 + temp_norm * 0.3
                            
            except (subprocess.TimeoutExpired, Exception) as e:
                pass
        
        # If we only have device files, return minimal activity
        return 0.05  # Small baseline value indicating GPU present but can't read stats
    
    def get_detailed_stats(self) -> dict:
        """Get detailed GPU statistics for debugging"""
        if not self.available:
            return {"available": False}
        
        # Try NVML first
        if self.use_nvml and self.device_handle:
            try:
                name = pynvml.nvmlDeviceGetName(self.device_handle)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.device_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000  # mW to W
                except:
                    power = "N/A"
                
                return {
                    "available": True,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "gpu_util": f"{util_rates.gpu}%",
                    "mem_util": f"{util_rates.memory}%",
                    "temperature": f"{temp} C",
                    "power": f"{power} W" if power != "N/A" else power,
                    "mem_used": f"{mem_info.used // (1024*1024)} MiB",
                    "mem_total": f"{mem_info.total // (1024*1024)} MiB",
                    "method": "NVML"
                }
            except Exception as e:
                pass
        
        # Fall back to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    parts = [p.strip() for p in output.split(',')]
                    if len(parts) >= 7:
                        return {
                            "available": True,
                            "name": parts[0],
                            "gpu_util": parts[1],
                            "mem_util": parts[2],
                            "temperature": parts[3],
                            "power": parts[4],
                            "mem_used": parts[5],
                            "mem_total": parts[6],
                            "method": "nvidia-smi"
                        }
        except:
            pass
            
        return {"available": self.available, "method": "device files only"}