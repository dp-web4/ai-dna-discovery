#!/usr/bin/env python3
"""
Test coherence engine with real hardware sensors.
Integrates vision, IMU, and persistent memory.
"""

import sys
import time
import logging
from pathlib import Path

# Add sensors directory to path
sys.path.insert(0, str(Path(__file__).parent / "sensors"))

from coherence_engine import (
    CoherenceEngine, Context, ContextState,
    TrustModel, RelevanceModel,
    CognitionSensor  # Still using simulated cognition for now
)
from sensors.real_vision_sensor import RealVisionSensor
from sensors.real_imu_sensor import RealIMUSensor
from sensors.persistent_memory_sensor import PersistentMemorySensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger("test_real_sensors")

def main():
    """Run coherence engine with real sensors."""
    
    logger.info("Initializing real sensors...")
    
    # Initialize sensors
    vision = RealVisionSensor()
    imu = RealIMUSensor()
    memory = PersistentMemorySensor(memory_dir=Path("memory"))
    cognition = CognitionSensor()  # Still simulated for now
    
    # Set up trust model with initial trust
    trust = TrustModel(
        base={
            "vision": 0.7,    # Start with decent trust in vision
            "imu": 0.6,       # Moderate trust in IMU
            "memory": 0.5,    # Build trust in memory over time
            "cognition": 0.4  # Lower initial trust in predictions
        }
    )
    
    # Set up relevance model with context-specific weights
    relevance = RelevanceModel(
        priors={
            # Stable context - rely on vision
            (ContextState.STABLE, "vision"): 0.8,
            (ContextState.STABLE, "imu"): 0.3,
            (ContextState.STABLE, "memory"): 0.4,
            (ContextState.STABLE, "cognition"): 0.2,
            
            # Moving context - boost IMU
            (ContextState.MOVING, "vision"): 0.6,
            (ContextState.MOVING, "imu"): 0.9,
            (ContextState.MOVING, "memory"): 0.3,
            (ContextState.MOVING, "cognition"): 0.3,
            
            # Unstable - balance all sensors
            (ContextState.UNSTABLE, "vision"): 0.5,
            (ContextState.UNSTABLE, "imu"): 0.6,
            (ContextState.UNSTABLE, "memory"): 0.6,
            (ContextState.UNSTABLE, "cognition"): 0.7,
            
            # Novel - rely on memory and prediction
            (ContextState.NOVEL, "vision"): 0.4,
            (ContextState.NOVEL, "imu"): 0.4,
            (ContextState.NOVEL, "memory"): 0.8,
            (ContextState.NOVEL, "cognition"): 0.9,
        }
    )
    
    # Create context
    ctx = Context(
        state=ContextState.STABLE,
        trust=trust,
        relevance=relevance
    )
    
    # Create engine
    engine = CoherenceEngine(
        sensors=[vision, imu, memory, cognition],
        context=ctx
    )
    
    logger.info("Starting coherence engine with real sensors...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        tick = 0
        while True:
            # Run engine step
            field_value = engine.step(tick=tick)
            
            # Get raw sensor readings for memory
            raw = {
                "vision": vision.read(tick=tick),
                "imu": imu.read(tick=tick),
                "memory": memory.read(tick=tick),
                "cognition": cognition.read(tick=tick)
            }
            
            # Update memory with experience
            current_state = engine.context.state.name
            trigger = None
            if engine.context.history:
                latest = engine.context.history[-1]
                trigger = latest.trigger
                
            memory.observe(
                context_state=current_state,
                sensor_readings=raw,
                field_value=field_value,
                trigger=trigger
            )
            
            # Log status every 10 ticks
            if tick % 10 == 0:
                logger.info(f"Tick {tick}: Field={field_value:.3f}, "
                          f"Context={current_state}, "
                          f"V={raw['vision']:.2f}, I={raw['imu']:.2f}, "
                          f"M={raw['memory']:.2f}, C={raw['cognition']:.2f}")
                
                # Get memory insights
                insights = memory.get_insights()
                logger.info(f"Memory: {insights['total_experiences']} experiences, "
                          f"stability={insights['stability']:.2f}, "
                          f"triggers={insights['trigger_rate']:.2f}")
                
                # Check IMU raw data if available
                if hasattr(imu, 'get_raw_data'):
                    imu_raw = imu.get_raw_data()
                    if imu_raw:
                        logger.debug(f"IMU raw: {imu_raw}")
                        
            tick += 1
            time.sleep(0.1)  # 10 Hz update rate
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        
        # Export history
        history_file = Path("coherence_history_real.json")
        engine.export_history(history_file)
        logger.info(f"Exported history to {history_file}")
        
        # Save final memory state
        final_insights = memory.get_insights()
        logger.info(f"Final memory state: {final_insights}")
        
    finally:
        # Cleanup
        del vision
        del imu
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()