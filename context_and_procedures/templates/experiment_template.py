"""
Standard Experiment Template

This template provides a consistent structure for all experiments,
ensuring proper logging, error handling, and result storage.
"""

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


class BaseExperiment:
    """Base class for all experiments."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.start_time = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up directories
        self.base_dir = Path(f"results/{self.name}_{self.timestamp}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.base_dir / "experiment.log"
        self.log(f"Initialized experiment: {self.name}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Console output
        print(log_entry)
        
        # File output
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
            
    def setup(self):
        """
        Set up experiment resources.
        Override this method for experiment-specific setup.
        """
        self.log("Running default setup")
        
    def validate_inputs(self):
        """
        Validate experiment inputs and configuration.
        Override for specific validation logic.
        """
        self.log("Validating inputs")
        
        # Example validation
        required_keys = ['seed', 'iterations']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
                
    def run_experiment(self) -> Dict[str, Any]:
        """
        Main experiment logic.
        Override this method with actual experiment code.
        
        Returns:
            Dictionary of results
        """
        raise NotImplementedError("Subclasses must implement run_experiment")
        
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze experimental results.
        Override for custom analysis.
        
        Args:
            results: Raw experimental results
            
        Returns:
            Analyzed results
        """
        self.log("Running default analysis")
        
        # Example analysis
        analysis = {
            'summary_statistics': {
                'mean': np.mean(results.get('values', [])),
                'std': np.std(results.get('values', [])),
                'min': np.min(results.get('values', [])),
                'max': np.max(results.get('values', []))
            }
        }
        
        return analysis
        
    def create_visualizations(self, results: Dict[str, Any]):
        """
        Create visualizations from results.
        Override for custom visualizations.
        """
        self.log("Creating visualizations")
        
        # Example visualization
        if 'values' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results['values'])
            plt.title(f"{self.name} Results")
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.savefig(self.base_dir / "results_plot.png", dpi=300)
            plt.close()
            
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        output_file = self.base_dir / "results.json"
        
        # Add metadata
        results_with_metadata = {
            'experiment_name': self.name,
            'timestamp': self.timestamp,
            'duration_seconds': time.time() - self.start_time,
            'config': self.config,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
            
        self.log(f"Results saved to {output_file}")
        
    def cleanup(self):
        """
        Clean up experiment resources.
        Override for specific cleanup needs.
        """
        self.log("Running cleanup")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.
        
        Returns:
            Complete results dictionary
        """
        self.start_time = time.time()
        
        try:
            # Setup
            self.log("Starting experiment")
            self.setup()
            
            # Validation
            self.validate_inputs()
            
            # Run experiment
            self.log("Running main experiment")
            raw_results = self.run_experiment()
            
            # Analysis
            self.log("Analyzing results")
            analyzed_results = self.analyze_results(raw_results)
            
            # Combine results
            self.results = {
                'raw': raw_results,
                'analysis': analyzed_results
            }
            
            # Visualizations
            self.create_visualizations(self.results)
            
            # Save results
            self.save_results(self.results)
            
            # Cleanup
            self.cleanup()
            
            # Success
            duration = time.time() - self.start_time
            self.log(f"Experiment completed successfully in {duration:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            # Log error
            self.log(f"Experiment failed: {str(e)}", level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")
            
            # Save partial results if any
            if self.results:
                self.results['error'] = str(e)
                self.save_results(self.results)
                
            # Re-raise
            raise
            
        finally:
            # Always cleanup
            try:
                self.cleanup()
            except:
                pass


# Example concrete experiment implementation
class ExampleExperiment(BaseExperiment):
    """Example experiment showing how to use the template."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'seed': 42,
            'iterations': 100,
            'learning_rate': 0.01
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
            
        super().__init__("example_experiment", default_config)
        
    def setup(self):
        """Set up experiment-specific resources."""
        super().setup()
        
        # Set random seed
        np.random.seed(self.config['seed'])
        
        # Initialize any models, data, etc.
        self.log("Setting up example experiment")
        
    def run_experiment(self) -> Dict[str, Any]:
        """Run the actual experiment."""
        results = {
            'values': [],
            'metrics': {}
        }
        
        # Simulate experiment
        for i in range(self.config['iterations']):
            # Simulate some computation
            value = np.random.randn() * (1 - i / self.config['iterations'])
            results['values'].append(value)
            
            # Log progress
            if i % 10 == 0:
                self.log(f"Progress: {i}/{self.config['iterations']}")
                
        # Calculate final metrics
        results['metrics']['final_value'] = results['values'][-1]
        results['metrics']['convergence_rate'] = abs(results['values'][-1] - results['values'][0])
        
        return results
        
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experimental results."""
        analysis = super().analyze_results(results)
        
        # Add custom analysis
        values = results['values']
        analysis['convergence'] = {
            'converged': abs(values[-1]) < 0.1,
            'iterations_to_convergence': self._find_convergence_point(values)
        }
        
        return analysis
        
    def _find_convergence_point(self, values: list, threshold: float = 0.1) -> Optional[int]:
        """Find iteration where values converged."""
        for i, value in enumerate(values):
            if abs(value) < threshold:
                return i
        return None


# Usage example
if __name__ == "__main__":
    # Create and run experiment
    config = {
        'seed': 123,
        'iterations': 50,
        'learning_rate': 0.1
    }
    
    experiment = ExampleExperiment(config)
    results = experiment.run()
    
    print(f"\nExperiment completed!")
    print(f"Final value: {results['raw']['metrics']['final_value']:.4f}")
    print(f"Converged: {results['analysis']['convergence']['converged']}")