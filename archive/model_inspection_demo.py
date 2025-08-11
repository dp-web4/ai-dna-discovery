#!/usr/bin/env python3
"""
Model Inspection and GPU Memory Analysis Demo
Tools for analyzing model weights, memory usage, and optimization
"""

import torch
import torch.nn as nn
from torchinfo import summary
from torchsummary import summary as summary_old
import onnx
import time
import psutil
import gpustat
from transformers import AutoModel, AutoTokenizer
import sys
import os

class ModelInspector:
    """Comprehensive model inspection and analysis tools"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    
    def gpu_memory_stats(self):
        """Get detailed GPU memory statistics"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
        
        print("\n=== GPU Memory Stats ===")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Max Cached: {torch.cuda.max_memory_reserved(0) / 1024**2:.2f} MB")
        
        # Using gpustat for more details
        gpus = gpustat.GPUStatCollection.new_query()
        for gpu in gpus:
            print(f"\nGPU {gpu.index}: {gpu.name}")
            print(f"Memory: {gpu.memory_used}MB / {gpu.memory_total}MB")
            print(f"Temperature: {gpu.temperature}°C")
            print(f"Utilization: {gpu.utilization}%")
    
    def inspect_model_weights(self, model):
        """Analyze model weights statistics"""
        print("\n=== Model Weight Analysis ===")
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            
            # Weight statistics
            if param.dim() > 1:  # Skip biases
                print(f"\nLayer: {name}")
                print(f"  Shape: {list(param.shape)}")
                print(f"  Mean: {param.mean().item():.6f}")
                print(f"  Std: {param.std().item():.6f}")
                print(f"  Min: {param.min().item():.6f}")
                print(f"  Max: {param.max().item():.6f}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")
    
    def profile_memory_usage(self, model, input_shape):
        """Profile memory usage during forward pass"""
        if not torch.cuda.is_available():
            print("CUDA required for memory profiling")
            return
        
        model = model.to(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        print("\n=== Memory Profiling ===")
        # Before forward pass
        start_mem = torch.cuda.memory_allocated()
        
        # Forward pass with timing
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        # After forward pass
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        
        print(f"Memory before forward: {start_mem / 1024**2:.2f} MB")
        print(f"Memory after forward: {end_mem / 1024**2:.2f} MB")
        print(f"Peak memory usage: {peak_mem / 1024**2:.2f} MB")
        print(f"Memory used by forward pass: {(end_mem - start_mem) / 1024**2:.2f} MB")
        print(f"Inference time: {inference_time * 1000:.2f} ms")
    
    def export_to_onnx(self, model, input_shape, output_path):
        """Export model to ONNX format"""
        model.eval()
        model = model.cpu()  # Ensure model is on CPU for ONNX export
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"\nModel exported to {output_path}")
        
        # Print ONNX model info
        print(f"ONNX model size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
    def save_model_checkpoint(self, model, optimizer, epoch, path):
        """Save model checkpoint with all necessary info"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        }
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved to {path}")
    
    def load_and_inspect_transformer(self, model_name="microsoft/phi-2"):
        """Load and inspect a transformer model"""
        print(f"\n=== Loading Transformer Model: {model_name} ===")
        
        # Monitor loading process
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Load model
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        
        if torch.cuda.is_available():
            model = model.to(self.device)
            end_mem = torch.cuda.memory_allocated()
            print(f"Model loaded. GPU memory used: {(end_mem - start_mem) / 1024**2:.2f} MB")
        
        # Use torchinfo for detailed summary
        print("\n=== Model Architecture ===")
        summary(model, device=self.device, verbose=0)
        
        return model


# Demo usage
if __name__ == "__main__":
    import os
    
    inspector = ModelInspector()
    
    # Show initial GPU state
    inspector.gpu_memory_stats()
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create and analyze model
    model = SimpleModel()
    
    # Inspect weights
    inspector.inspect_model_weights(model)
    
    # Profile memory usage
    input_shape = (1, 3, 32, 32)  # batch_size, channels, height, width
    inspector.profile_memory_usage(model, input_shape)
    
    # Show model summary using torchinfo
    print("\n=== Model Summary (torchinfo) ===")
    summary(model, input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Export to ONNX
    onnx_path = "/home/dp/ai-workspace/ai-agents/simple_model.onnx"
    inspector.export_to_onnx(model, input_shape, onnx_path)
    
    # Save checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint_path = "/home/dp/ai-workspace/ai-agents/model_checkpoint.pth"
    inspector.save_model_checkpoint(model, optimizer, epoch=1, path=checkpoint_path)
    
    # Optional: Load and inspect a small transformer model
    # Uncomment to test with a real transformer model
    # inspector.load_and_inspect_transformer("microsoft/phi-2")
    
    # Final GPU stats
    print("\n=== Final GPU State ===")
    inspector.gpu_memory_stats()