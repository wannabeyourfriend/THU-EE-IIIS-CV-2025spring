import numpy as np
from convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img
from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class CannyConfig:
    """Configuration class for Canny edge detection parameters"""
    # NMS parameters
    parallel_distance: int = 2
    angle_tolerance: float = 20.0
    local_max_threshold: float = 0.95
    parallel_mag_threshold: float = 0.9
    neighbor_similarity_threshold: float = 0.05
    secondary_mag_ratio: float = 0.7
    
    # Hysteresis parameters
    low_ratio: float = 0.12
    high_ratio: float = 0.22
    
    # Edge connectivity parameters
    min_connected_edges: int = 2
    
    @classmethod
    def get_preset_configs(cls) -> dict:
        """Return preset configurations for different scenarios"""
        return {
            'default': cls(),
            'fine_detail': cls(
                parallel_distance=1,
                angle_tolerance=15.0,
                local_max_threshold=0.98,
                parallel_mag_threshold=0.95,
                neighbor_similarity_threshold=0.01,
                low_ratio=0.10,
                high_ratio=0.35
            ),
            'strong_edges': cls(
                parallel_distance=3,
                angle_tolerance=25.0,
                local_max_threshold=0.90,
                parallel_mag_threshold=0.85,
                neighbor_similarity_threshold=0.08,
                low_ratio=0.10,
                high_ratio=0.20
            ),
            'parallel_emphasis': cls(
                parallel_distance=2,
                angle_tolerance=15.0,
                local_max_threshold=0.92,
                parallel_mag_threshold=0.88,
                neighbor_similarity_threshold=0.04,
                secondary_mag_ratio=0.65,
                low_ratio=0.14,
                high_ratio=0.24
            )
        }

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(np.square(x_grad) + np.square(y_grad))
    
    direction_grad = np.arctan2(y_grad, x_grad)
    
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir, config: CannyConfig):
    """Modified NMS function to use configuration parameters"""
    height, width = grad_mag.shape
    
    NMS_output = np.zeros_like(grad_mag)
    
    # Convert angles to degrees and normalize to [0,180)
    angle = np.rad2deg(grad_dir) % 180
    
    # Define distance threshold for parallel edge detection
    parallel_distance = config.parallel_distance
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            current_angle = angle[i, j]
            current_mag = grad_mag[i, j]
            
            # Calculate interpolation weight based on angle
            weight = current_angle % 45 / 45
            
            # Determine gradient direction and neighbors
            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                # Horizontal direction (0 degrees)
                neighbor1 = grad_mag[i, j+1]
                neighbor2 = grad_mag[i, j-1]
                # Check for parallel edges in vertical direction
                parallel_dir = [(i-k, j) for k in range(1, parallel_distance+1) if i-k >= 0] + \
                              [(i+k, j) for k in range(1, parallel_distance+1) if i+k < height]
            elif 22.5 <= current_angle < 67.5:
                # 45 degrees direction
                neighbor1 = grad_mag[i-1, j+1] * (1-weight) + grad_mag[i-1, j] * weight
                neighbor2 = grad_mag[i+1, j-1] * (1-weight) + grad_mag[i+1, j] * weight
                # Check for parallel edges in -45 degrees direction
                parallel_dir = [(i-k, j-k) for k in range(1, parallel_distance+1) if i-k >= 0 and j-k >= 0] + \
                              [(i+k, j+k) for k in range(1, parallel_distance+1) if i+k < height and j+k < width]
            elif 67.5 <= current_angle < 112.5:
                # Vertical direction (90 degrees)
                neighbor1 = grad_mag[i-1, j]
                neighbor2 = grad_mag[i+1, j]
                # Check for parallel edges in horizontal direction
                parallel_dir = [(i, j-k) for k in range(1, parallel_distance+1) if j-k >= 0] + \
                              [(i, j+k) for k in range(1, parallel_distance+1) if j+k < width]
            else:  # 112.5 <= current_angle < 157.5
                # -45 degrees direction
                neighbor1 = grad_mag[i-1, j-1] * (1-weight) + grad_mag[i-1, j] * weight
                neighbor2 = grad_mag[i+1, j+1] * (1-weight) + grad_mag[i+1, j] * weight
                # Check for parallel edges in 45 degrees direction
                parallel_dir = [(i-k, j+k) for k in range(1, parallel_distance+1) if i-k >= 0 and j+k < width] + \
                              [(i+k, j-k) for k in range(1, parallel_distance+1) if i+k < height and j-k >= 0]
            
            # Primary NMS check - is this pixel a local maximum in gradient direction?
            if current_mag > max(neighbor1, neighbor2):
                NMS_output[i, j] = current_mag
            elif (abs(current_mag - neighbor1) < config.neighbor_similarity_threshold * current_mag or 
                  abs(current_mag - neighbor2) < config.neighbor_similarity_threshold * current_mag):
                local_window = grad_mag[max(0,i-1):min(height,i+2), max(0,j-1):min(width,j+2)]
                if current_mag >= config.local_max_threshold * np.max(local_window):
                    NMS_output[i, j] = current_mag
            
            # Secondary check - is this part of a parallel edge structure?
            elif current_mag > config.secondary_mag_ratio * max(neighbor1, neighbor2):
                parallel_edge_found = False
                for pi, pj in parallel_dir:
                    # Check if there's a significant gradient in the parallel position
                    if grad_mag[pi, pj] > config.parallel_mag_threshold * current_mag:
                        if abs(angle[pi, pj] - current_angle) % 180 < config.angle_tolerance:
                            parallel_edge_found = True
                            break
                
                if parallel_edge_found:
                    NMS_output[i, j] = current_mag
    
    return NMS_output

def hysteresis_thresholding(img, config: CannyConfig):
    """Modified hysteresis function to use configuration parameters"""
    height, width = img.shape
    
    output = np.zeros_like(img)
    
    # Adjust threshold ratios for better edge detection
    low_ratio = config.low_ratio
    high_ratio = config.high_ratio
    
    max_val = np.max(img)
    
    high_threshold = max_val * high_ratio
    low_threshold = max_val * low_ratio
    
    # Identify strong and weak edges
    strong_edges = (img >= high_threshold)
    weak_edges = ((img >= low_threshold) & (img < high_threshold))
    
    # Mark strong edges in the output
    output[strong_edges] = 1
    
    # 8-connected neighborhood directions
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    # Use BFS instead of DFS to avoid stack overflow for large images
    from collections import deque
    
    # Initialize queue with all strong edge pixels
    queue = deque()
    for i in range(height):
        for j in range(width):
            if strong_edges[i, j]:
                # Add all adjacent weak edges to the queue
                for k in range(8):
                    ni, nj = i + dx[k], j + dy[k]
                    if 0 <= ni < height and 0 <= nj < width:
                        if weak_edges[ni, nj] and output[ni, nj] == 0:
                            queue.append((ni, nj))
                            output[ni, nj] = 1  # Mark as visited
    
    # Process the queue
    while queue:
        i, j = queue.popleft()
        
        # Check all neighbors
        for k in range(8):
            ni, nj = i + dx[k], j + dy[k]
            if 0 <= ni < height and 0 <= nj < width:
                if weak_edges[ni, nj] and output[ni, nj] == 0:
                    output[ni, nj] = 1  # Mark as edge
                    queue.append((ni, nj))  # Add to queue
    
    # Apply a final pass to enhance connectivity of parallel edges
    for i in range(1, height-1):
        for j in range(1, width-1):
            if output[i, j] == 0 and weak_edges[i, j]:
                # Count adjacent edge pixels
                edge_count = 0
                for k in range(8):
                    ni, nj = i + dx[k], j + dy[k]
                    if 0 <= ni < height and 0 <= nj < width and output[ni, nj] == 1:
                        edge_count += 1
                
                # If this weak edge connects multiple edge segments, include it
                if edge_count >= config.min_connected_edges:
                    output[i, j] = 1
    
    return output



def process_image(input_path: str, config: CannyConfig, output_suffix: str = '') -> None:
    """Process image with given configuration"""
    # Create result directory if not exists
    os.makedirs('result', exist_ok=True)
    
    # Load and preprocess image
    input_img = read_img(input_path)/255
    blur_img = Gaussian_filter(input_img)
    
    # Compute gradients
    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)
    
    # Compute magnitude and direction
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)
    
    # Apply NMS and hysteresis with given configuration
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad, config)
    output_img = hysteresis_thresholding(NMS_output, config)
    
    # Save result
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"result/Canny/Canny_{base_name}_{output_suffix}.png"
    write_img(output_path, output_img*255)
    return output_img

if __name__=="__main__":
    # Get all preset configurations
    configs = CannyConfig.get_preset_configs()
    
    # Process image with all configurations
    input_path = "Lenna.png"
    results = {}
    
    for config_name, config in configs.items():
        print(f"Processing with {config_name} configuration...")
        results[config_name] = process_image(input_path, config, config_name)
    
    # Optional: Add code to evaluate results
    # For example, you could compare edge continuity, number of detected edges, etc.
