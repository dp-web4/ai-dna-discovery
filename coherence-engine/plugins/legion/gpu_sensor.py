"""
GPU sensor for Legion (RTX 4090) - monitors GPU utilization and temperature
"""
import subprocess
import re
from typing import Optional

class GPUSensor:
    """GPU utilization and temperature sensor using nvidia-smi"""
    
    def __init__(self):
        self.id = "gpu"
        self.available = self.check_nvidia_smi()
        self.last_util = 0.0
        self.last_temp = 0.0
        
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
            print(f"GPU sensor error: {e}")
            
        return 0.0
    
    def get_detailed_stats(self) -> dict:
        """Get detailed GPU statistics for debugging"""
        if not self.available:
            return {"available": False}
            
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
                            "mem_total": parts[6]
                        }
        except:
            pass
            
        return {"available": False}