from PIL import Image
import os
import sys
# -1: CPU Only execution - 0: GPU only if compatible
# Newer versions of SionnaRT use Dr.Jit which requires your GPU to have a Compute Capability (SM) > 7.0
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # In my case, my GPU is not supported so I have to rely on the CPU (slower)
# Log level of Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib;
matplotlib.use('Agg') # Use Agg as backend instead of a GUI one (TkAgg) since we render on a file and dont need a GUI toolkit
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import json
import argparse
import logging
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, PathSolver

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def setup_gpu(gpu_num=0):
  """Configures GPU for TensorFlow and suppresses warnings."""
  try:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
      logger.warning("No visible GPU found by TensorFlow.")
      return
    logger.info(f"Found visible GPU(s): {gpus}")
    configured_any = False
    for gpu in gpus: 
      try:
        tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Enabled memory growth for {gpu.name}")
        configured_any = True
      except RuntimeError as e:
        logger.warning(f"Could not set memory growth for {gpu.name} (might be already initialized or other issue): {e}")
    if not configured_any:
      logger.warning("Memory growth could not be configured for any GPU.")

  except Exception as e:
    logger.error(f"An unexpected error occurred during GPU memory setup: {e}", exc_info=True)
  tf.get_logger().setLevel('ERROR')

def parse_arguments():
  """Parses command-line arguments for receiver position and orientation."""
  parser = argparse.ArgumentParser(description="Render scene with dynamic position and orientation.")
  # Arguments are required for the script to function
  parser.add_argument('--position', type=str, required=True, help="Position of the receiver (JSON formatted list, e.g., '[0, -30, 0.1]')")
  parser.add_argument('--orientation', type=str, required=True, help="Orientation of the receiver (JSON formatted list, e.g., '[0, 0, 0]')")
  
  args = parser.parse_args()
  logger.info("Parsing arguments...")
  
  try:
    # Convert JSON strings to Python lists
    rx_pos = json.loads(args.position)
    rx_ori = json.loads(args.orientation)
    
    # Basic validation: Check if inputs are lists of numbers
    if not isinstance(rx_pos, list) or not all(isinstance(x, (int, float)) for x in rx_pos) or len(rx_pos) != 3:
        raise ValueError("Position must be a JSON list of 3 numbers.")
    if not isinstance(rx_ori, list) or not all(isinstance(x, (int, float)) for x in rx_ori) or len(rx_ori) != 3:
        raise ValueError("Orientation must be a JSON list of 3 numbers.")
        
    logger.info(f"Successfully parsed position: {rx_pos} and orientation: {rx_ori}")
    return rx_pos, rx_ori
  except json.JSONDecodeError as e:
    logger.error(f"Failed to decode JSON arguments. Position: '{args.position}', Orientation: '{args.orientation}'. Error: {e}", exc_info=True)
    raise ValueError(f"Invalid JSON format in arguments: {e}") from e
  except ValueError as e:
    logger.error(f"Invalid argument value: {e}", exc_info=True)
    raise

def load_scene_from_xml(scene_filename="povo_scene.xml"):
  """Loads scene from XML file located relative to the project structure."""
  filepath = None
  try:
    # Get absolute path for scene.xml file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir) 
    scene_dir = os.path.join(src_dir, "scene")
    filepath = os.path.join(scene_dir, scene_filename)
    
    logger.info(f"Attempting to load scene from: {filepath}")
    scene = load_scene(filepath)
    logger.info(f"Scene '{filepath}' loaded successfully.")
    return scene
  except FileNotFoundError:
    logger.error(f"Scene file not found at calculated path: {filepath}", exc_info=True)
    raise
  except Exception as e: 
    logger.error(f"Failed to load scene from {filepath}: {e}", exc_info=True)
    raise

def setup_camera():
  """Initializes and returns a camera with pre-defined position and focus."""
  try:
    camera = Camera(position=[22, 47, 110], look_at=[-18, 0, -8])
    camera.name = "def_camera"
    logger.info("Camera object created successfully.")
    return camera
  except Exception as e:
    logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
    raise

def configure_antenna_arrays(scene):
  """Configures transmitter and receiver antenna arrays on the scene."""
  # Antenna parameters (rows, cols, spacing, pattern, polarization) significantly affect simulation results.
  logger.info("Configuring TX/RX antenna arrays.")

  scene.tx_array = PlanarArray(
    num_rows=4, num_cols=4, # 4x4 array
    vertical_spacing=0.5, horizontal_spacing=0.5, # Spacing in wavelengths
    pattern="tr38901",  # Standard 3GPP antenna pattern
    polarization="V"  # Vertical polarization
  )

  scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1, # Single element
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="dipole", # Simple dipole pattern
    polarization="cross"  # Cross-polarized to capture different signal components
  )
  logger.info("Antenna arrays configured.")

def add_transmitter_receiver(scene, rx_pos, rx_ori):
  """Adds and configures the transmitter and receiver objects in the scene."""
  logger.info(f"Adding TX at fixed position and RX at {rx_pos} with orientation {rx_ori}.")
  tx = Transmitter("tx", [-39, -15, 4])
  scene.add(tx)

  rx = Receiver("rx", rx_pos, rx_ori)
  scene.add(rx)

  tx.look_at(rx)  # Make the transmitter point at the receiver
  scene.frequency = 2.14e9  # Set frequency in Hz
  scene.synthetic_array = True  # Enable synthetic array for faster ray tracing
  logger.info(f"Scene frequency set to {(scene.frequency / 1e9).numpy().item():.2f} GHz.")

def compute_paths(scene, max_depth=5, num_samples=1e6):
  """Computes the propagation paths using Sionna RT."""
  # This is often the most computationally intensive part, especially on CPU.
  # max_depth: Controls reflection/diffraction depth. Higher = more realistic but significantly slower.
  # num_samples: Number of rays cast from TX. Higher = better chance of finding paths, more accurate results, but slower.
  logger.info(f"Computing paths with max_depth={max_depth}, num_samples={num_samples:.1e}...")
  paths = None
  try:
    samples_int = int(num_samples)
    solver = PathSolver()
    paths = solver(scene, max_depth=max_depth, samples_per_src=samples_int)
    logger.info(f"Path computation finished.")
    return paths
  except Exception as e:
    logger.error(f"CRITICAL: Error during path computation: {e}", exc_info=True)
    raise

def get_next_filename(directory, base_name, extension):
  """Finds the next available filename in the directory, with an incremented numeric suffix."""
  i = 1
  while True:
    filename = f"{base_name}_{i}.{extension}"
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
      return file_path
    i += 1

def render_and_save(scene, paths=None, camera=None, resolution=[480, 320]):
  """Renders the scene using matplotlib and saves the output image with a unique filename."""
  logger.info(f"Starting scene rendering.")
  try:
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Create the "renders" directory if it doesn't exist
    renders_directory = os.path.join(current_directory, "renders")
    os.makedirs(renders_directory, exist_ok=True)
    # Automatically generate a unique filename with an incremental number
    output_path = get_next_filename(renders_directory, "paths_render", "png")
    render_kwargs = {
      "camera": camera,
      "filename": output_path,
      "paths": paths, # Pass computed paths to visualize them
      "show_devices": True, # Show TX and RX locations
      "num_samples": 512, # Rendering parameters - adjust for quality vs. speed trade-off.
      "resolution": resolution # Image dimensions [width, height]
    }

    logger.info(f"Rendering scene...")
    scene.render_to_file(**render_kwargs)
    logger.info(f"Saving rendered image to {output_path}...")

  except Exception as e:
    error_msg = f"CRITICAL: An error occurred during rendering or saving"
    if output_path:
      error_msg += f" to {output_path}"
    error_msg += f": {e}"
    logger.error(error_msg, exc_info=True)
    sys.exit(1)

# Main execution
def main():
  logger.info("--- Starting SionnaRT Simulation Script ---")
  if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
      logger.info("CUDA_VISIBLE_DEVICES set to '-1'. Forcing CPU execution.")
  else:
      logger.info(f"CUDA_VISIBLE_DEVICES set to '{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}'. Attempting GPU execution if available.")
    # 0. If CUDA_VISIBLE_DEVICES IS NOT '-1' and you want to use the GPU for execution, then
    # uncomment the following line if you have a compatible NVIDIA GPU with the CUDA and cuDNN drivers, properly installed, to greatly accelerate calculations:
    # setup_gpu()
  try:
    # 1. Parse arguments
    rx_pos, rx_ori = parse_arguments()

    # 2. Load scene and set camera
    logger.info("Loading scene...")
    scene = load_scene_from_xml("povo_scene.xml")
    logger.info("Setting up camera...")
    camera_item = setup_camera()

    # 3. Configure antennas and add transmitter and dynamic receiver
    logger.info("Configuring antennas...")
    configure_antenna_arrays(scene)
    logger.info("Adding transmitter and receiver...")
    add_transmitter_receiver(scene, rx_pos, rx_ori)

    # 4. Compute paths
    logger.info("Computing paths...")
    paths = compute_paths(scene)

    # 5. Render and save the scene
    logger.info("Rendering and saving scene...")
    render_and_save(scene, paths=paths, camera=camera_item, resolution=[480, 320])
    logger.info("--- SionnaRT script finished successfully. ---")
    sys.exit(0)

  except (ValueError, FileNotFoundError, RuntimeError, TypeError, AttributeError) as e: 
    logger.critical(f"A critical error occurred during script execution: {e}", exc_info=True)
    sys.exit(1)
  except Exception as e:
    logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    sys.exit(1)

if __name__ == "__main__":
  main()
