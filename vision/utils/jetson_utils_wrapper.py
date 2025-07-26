#!/usr/bin/env python3
"""
Wrapper for jetson-utils to simplify GPU vision operations
Provides easy interface for camera input and display output with GPU buffers
"""

try:
    import jetson.utils
    import jetson.inference
    JETSON_UTILS_AVAILABLE = True
except ImportError:
    JETSON_UTILS_AVAILABLE = False
    print("jetson-utils not installed. Install from: https://github.com/dusty-nv/jetson-inference")

class JetsonVision:
    def __init__(self, camera_device="/dev/video0", display_type="display://0"):
        """
        Initialize Jetson vision pipeline
        
        Args:
            camera_device: Camera source (CSI, V4L2, RTSP, etc.)
            display_type: Output display ("display://0" or "rtp://ip:port")
        """
        if not JETSON_UTILS_AVAILABLE:
            raise ImportError("jetson-utils required but not installed")
            
        self.camera = None
        self.display = None
        self.camera_device = camera_device
        self.display_type = display_type
        
    def init_camera(self, width=1280, height=720, framerate=30):
        """Initialize camera with specified resolution"""
        # For CSI camera
        if self.camera_device == "/dev/video0":
            camera_str = f"csi://0"  # Use CSI camera 0
        else:
            camera_str = self.camera_device
            
        self.camera = jetson.utils.videoSource(
            camera_str,
            argv=[
                f"--input-width={width}",
                f"--input-height={height}", 
                f"--input-rate={framerate}"
            ]
        )
        
        print(f"Camera initialized: {width}x{height} @ {framerate}fps")
        
    def init_display(self):
        """Initialize display output"""
        self.display = jetson.utils.videoOutput(self.display_type)
        print(f"Display initialized: {self.display_type}")
        
    def capture_frame(self):
        """
        Capture frame from camera (stays in GPU memory)
        Returns CUDA image object
        """
        cuda_img = self.camera.Capture()
        
        if cuda_img is None:
            return None
            
        # Image is already in GPU memory (CUDA)
        # Properties: width, height, format, etc.
        return cuda_img
        
    def display_frame(self, cuda_img, overlay=None):
        """
        Display frame (from GPU memory)
        
        Args:
            cuda_img: CUDA image from capture
            overlay: Optional overlay (detections, text, etc.)
        """
        if overlay:
            # Overlay rendering happens on GPU
            self.display.Render(cuda_img, overlay)
        else:
            self.display.Render(cuda_img)
            
    def get_cuda_buffer(self, cuda_img):
        """
        Get raw CUDA memory pointer for custom GPU operations
        
        Returns:
            (ptr, size, format, width, height)
        """
        return (
            cuda_img.cudaPtr,
            cuda_img.size,
            cuda_img.format,
            cuda_img.width,
            cuda_img.height
        )
        
    def simple_loop(self, process_func=None):
        """
        Simple capture-process-display loop
        
        Args:
            process_func: Optional function to process each frame
                         Should take cuda_img and return overlay
        """
        while True:
            # Capture (GPU)
            cuda_img = self.capture_frame()
            
            if cuda_img is None:
                continue
                
            overlay = None
            
            # Process if function provided
            if process_func:
                overlay = process_func(cuda_img)
                
            # Display (GPU)
            self.display_frame(cuda_img, overlay)
            
            # Check for exit
            if not self.display.IsStreaming():
                break
                
        print("Vision loop ended")

# Example usage functions
def example_edge_detection(cuda_img):
    """Example: Run edge detection on GPU"""
    # This would use CUDA kernels for edge detection
    # For now, just return None (no overlay)
    return None

def example_inference(model):
    """Returns a process function that runs inference"""
    def process(cuda_img):
        # Run detection (GPU)
        detections = model.Detect(cuda_img)
        return detections
    return process

# Demo script
if __name__ == "__main__":
    print("Jetson Vision Utils Demo")
    
    if not JETSON_UTILS_AVAILABLE:
        print("Please install jetson-utils first")
        print("git clone https://github.com/dusty-nv/jetson-inference")
        print("cd jetson-inference")
        print("docker/run.sh  # or build from source")
        sys.exit(1)
        
    # Create vision pipeline
    vision = JetsonVision()
    
    # Initialize camera and display
    vision.init_camera(1280, 720, 30)
    vision.init_display()
    
    print("Starting simple camera loop...")
    print("Close window to exit")
    
    # Run simple loop (no processing)
    vision.simple_loop()
    
    # For object detection:
    # net = jetson.inference.detectNet("ssd-mobilenet-v2")
    # vision.simple_loop(example_inference(net))