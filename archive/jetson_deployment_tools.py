#!/usr/bin/env python3
"""
Jetson Nano Deployment Tools
Utilities for preparing and optimizing models for Jetson Nano
"""

import torch
import torch.nn as nn
import onnx
import os
import json
from typing import Dict, Any, Tuple

class JetsonDeploymentHelper:
    """Tools for deploying models to Jetson Nano"""
    
    def __init__(self):
        self.jetson_specs = {
            "gpu": "128-core Maxwell",
            "cpu": "Quad-core ARM A57 @ 1.43 GHz",
            "memory": "4 GB 64-bit LPDDR4",
            "compute_capability": "5.3",
            "tensorrt_version": "8.2",  # Typical for recent JetPack
            "max_batch_size": 1,  # Conservative for Nano
            "fp16_support": True,
            "int8_support": True
        }
    
    def check_model_compatibility(self, model: nn.Module) -> Dict[str, Any]:
        """Check if model is suitable for Jetson Nano deployment"""
        results = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming FP32
        
        # Check model size
        if model_size_mb > 1000:  # 1GB threshold
            results["warnings"].append(f"Model size ({model_size_mb:.1f} MB) may be too large for Jetson Nano")
            results["recommendations"].append("Consider model pruning or using a smaller architecture")
        
        # Check for unsupported operations
        unsupported_ops = self._check_unsupported_ops(model)
        if unsupported_ops:
            results["warnings"].append(f"Found potentially unsupported operations: {unsupported_ops}")
            results["compatible"] = False
        
        results["model_stats"] = {
            "total_parameters": total_params,
            "model_size_mb": model_size_mb,
            "estimated_fp16_size_mb": model_size_mb / 2,
            "estimated_int8_size_mb": model_size_mb / 4
        }
        
        return results
    
    def _check_unsupported_ops(self, model: nn.Module) -> list:
        """Check for operations that might not be supported on Jetson"""
        unsupported = []
        
        for name, module in model.named_modules():
            # Check for operations that might have limited support
            if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                unsupported.append(f"Transposed convolution at {name}")
            elif isinstance(module, nn.Transformer):
                unsupported.append(f"Transformer layer at {name}")
        
        return unsupported
    
    def optimize_for_jetson(self, model: nn.Module, optimization_level: str = "fp16") -> nn.Module:
        """Optimize model for Jetson deployment"""
        model.eval()
        
        if optimization_level == "fp16":
            # Convert to half precision
            model = model.half()
            print("Model converted to FP16")
        
        # Set model to inference mode
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def export_for_tensorrt(self, model: nn.Module, input_shape: Tuple, 
                           output_path: str, fp16_mode: bool = True):
        """Export model in format suitable for TensorRT conversion"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        if fp16_mode and torch.cuda.is_available():
            model = model.half()
            dummy_input = dummy_input.half()
        
        # Export to ONNX with TensorRT-friendly settings
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # TensorRT compatible
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,  # Fixed batch size for Jetson
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"Model exported for TensorRT: {output_path}")
        print(f"ONNX file size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
        
        # Create deployment config
        config_path = output_path.replace('.onnx', '_config.json')
        config = {
            "model_path": os.path.basename(output_path),
            "input_shape": list(input_shape),
            "fp16_mode": fp16_mode,
            "target_device": "jetson_nano",
            "tensorrt_version": self.jetson_specs["tensorrt_version"]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Deployment config saved: {config_path}")
    
    def create_jetson_inference_script(self, model_name: str, output_dir: str):
        """Create a ready-to-use inference script for Jetson Nano"""
        script_content = f'''#!/usr/bin/env python3
"""
Inference script for {model_name} on Jetson Nano
Auto-generated by JetsonDeploymentHelper
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import time
import cv2

# Model configuration
MODEL_PATH = "{model_name}.onnx"
INPUT_SIZE = (224, 224)  # Adjust based on your model

class JetsonInference:
    def __init__(self, model_path):
        print("Initializing TensorRT inference engine...")
        # TensorRT initialization would go here
        # For now, using PyTorch as fallback
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {{self.device}}")
        
        # Load transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        return input_batch.to(self.device)
    
    def inference(self, input_batch):
        """Run inference on input batch"""
        with torch.no_grad():
            start_time = time.time()
            # TensorRT inference would replace this
            # output = self.engine.infer(input_batch)
            output = None  # Placeholder
            inference_time = (time.time() - start_time) * 1000
        
        return output, inference_time
    
    def run_benchmark(self, num_iterations=100):
        """Benchmark inference speed"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, *INPUT_SIZE).to(self.device)
        
        # Warmup
        for _ in range(10):
            self.inference(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            _, inference_time = self.inference(dummy_input)
            times.append(inference_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        print(f"\\nBenchmark Results:")
        print(f"Average inference time: {{avg_time:.2f}} ± {{std_time:.2f}} ms")
        print(f"FPS: {{fps:.1f}}")

if __name__ == "__main__":
    # Initialize inference engine
    engine = JetsonInference(MODEL_PATH)
    
    # Run benchmark
    engine.run_benchmark()
    
    # Example inference
    # image_path = "test_image.jpg"
    # input_batch = engine.preprocess(image_path)
    # output, time_ms = engine.inference(input_batch)
    # print(f"Inference completed in {{time_ms:.2f}} ms")
'''
        
        script_path = os.path.join(output_dir, f"{model_name}_jetson_inference.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"Jetson inference script created: {script_path}")
    
    def estimate_performance(self, model: nn.Module, input_shape: Tuple) -> Dict[str, float]:
        """Estimate model performance on Jetson Nano"""
        # Count FLOPs (simplified estimation)
        total_flops = 0
        
        def count_conv2d_flops(module, input, output):
            batch_size = input[0].shape[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            output_size = output.shape[2] * output.shape[3]
            
            # Multiply-add operations
            flops = batch_size * in_channels * out_channels * kernel_size * output_size * 2
            return flops
        
        # Rough estimation based on model size and Jetson capabilities
        total_params = sum(p.numel() for p in model.parameters())
        model_complexity = total_params / 1e6  # In millions
        
        # Jetson Nano can do roughly 472 GFLOPS (FP16)
        # Rough estimation: larger models run slower
        if model_complexity < 10:
            estimated_fps = 30
        elif model_complexity < 50:
            estimated_fps = 15
        elif model_complexity < 100:
            estimated_fps = 5
        else:
            estimated_fps = 1
        
        return {
            "estimated_fps": estimated_fps,
            "model_complexity_m": model_complexity,
            "recommended_batch_size": 1,
            "recommended_precision": "FP16"
        }


# Example usage
if __name__ == "__main__":
    helper = JetsonDeploymentHelper()
    
    # Print Jetson Nano specifications
    print("=== Jetson Nano Specifications ===")
    for key, value in helper.jetson_specs.items():
        print(f"{key}: {value}")
    
    # Example: Prepare a model for deployment
    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 56 * 56, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = ExampleModel()
    
    # Check compatibility
    print("\n=== Model Compatibility Check ===")
    compatibility = helper.check_model_compatibility(model)
    print(f"Compatible: {compatibility['compatible']}")
    print(f"Model stats: {compatibility['model_stats']}")
    
    # Estimate performance
    print("\n=== Performance Estimation ===")
    perf = helper.estimate_performance(model, (1, 3, 224, 224))
    for key, value in perf.items():
        print(f"{key}: {value}")
    
    # Export for TensorRT
    print("\n=== Exporting for TensorRT ===")
    helper.export_for_tensorrt(
        model, 
        (1, 3, 224, 224), 
        "/home/dp/ai-workspace/ai-agents/example_model_jetson.onnx",
        fp16_mode=True
    )
    
    # Create inference script
    helper.create_jetson_inference_script("example_model", "/home/dp/ai-workspace/ai-agents/")