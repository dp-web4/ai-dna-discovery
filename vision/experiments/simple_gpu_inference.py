#!/usr/bin/env python3
"""
Simple GPU Vision + Inference Pipeline for Jetson Orin Nano
Demonstrates camera → GPU inference → results with zero-copy buffers
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import time
import sys

# Initialize GStreamer
Gst.init(None)

class GPUVisionPipeline:
    def __init__(self):
        self.pipeline = None
        self.frame_count = 0
        self.start_time = time.time()
        
    def create_pipeline(self):
        """Create GStreamer pipeline with NVMM buffers"""
        pipeline_str = '''
            nvarguscamerasrc sensor-id=0 !
            video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 !
            nvvidconv !
            video/x-raw(memory:NVMM), format=NV12 !
            nvvidconv !
            video/x-raw, format=BGRx !
            videoconvert !
            video/x-raw, format=BGR !
            appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        '''
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get the appsink element
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.connect('new-sample', self.on_new_sample)
        
    def on_new_sample(self, sink):
        """Callback for new frames - runs inference here"""
        sample = sink.emit('pull-sample')
        if sample:
            # Get buffer
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get dimensions
            height = caps.get_structure(0).get_value('height')
            width = caps.get_structure(0).get_value('width')
            
            # Map buffer to numpy array (zero-copy if possible)
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # This is where GPU inference would happen
                # For now, just demonstrate we have the frame
                frame_data = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Simulate GPU inference
                self.simulate_inference(frame_data)
                
                buffer.unmap(map_info)
            
            self.frame_count += 1
            
            # Print FPS every 30 frames
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"FPS: {fps:.2f} | Frames: {self.frame_count}")
        
        return Gst.FlowReturn.OK
    
    def simulate_inference(self, frame):
        """Placeholder for GPU inference"""
        # In real implementation, this would:
        # 1. Keep data in GPU memory
        # 2. Run TensorRT/CUDA inference
        # 3. Return results without CPU copy
        
        # For now, just calculate mean (would be GPU kernel)
        # mean_value = np.mean(frame)  # Don't do this - it's CPU!
        
        # Simulate GPU processing time
        time.sleep(0.001)  # 1ms inference
        
    def run(self):
        """Start the pipeline"""
        self.create_pipeline()
        
        # Start playing
        self.pipeline.set_state(Gst.State.PLAYING)
        
        print("GPU Vision Pipeline Started")
        print("Press Ctrl+C to stop")
        
        # Run main loop
        try:
            loop = GLib.MainLoop()
            loop.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        
        # Clean up
        self.pipeline.set_state(Gst.State.NULL)
        
        # Print stats
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        print(f"\nTotal frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")

def main():
    print("🚀 Jetson GPU Vision Pipeline Demo")
    print("=" * 40)
    print("This demonstrates the camera → GPU pipeline")
    print("Real inference would use TensorRT/CUDA")
    print("=" * 40)
    
    pipeline = GPUVisionPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()