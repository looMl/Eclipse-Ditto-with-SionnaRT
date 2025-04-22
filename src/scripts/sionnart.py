import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
import logging
import sys

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# GPU Setup and TensorFlow Configuration
def setup_gpu(gpu_num=0):
  """Configures GPU for TensorFlow and suppresses warnings."""
  if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
      print(e)
  tf.get_logger().setLevel('ERROR')

# Check if running in Colab and adjust preview mode
def check_colab():
  try:
    import google.colab
    return True
  except:
    return os.getenv("SIONNA_NO_PREVIEW", False)

# Command-line argument parsing
def parse_arguments():
  parser = argparse.ArgumentParser(description="Render scene with dynamic position and orientation.")
  parser.add_argument('--position', type=str, help="Position of the receiver (JSON formatted list)")
  parser.add_argument('--orientation', type=str, help="Orientation of the receiver (JSON formatted list)")
  args = parser.parse_args()
  # Convert JSON strings to Python lists
  rx_pos = json.loads(args.position)
  rx_ori = json.loads(args.orientation)
  return rx_pos, rx_ori

# Scene and Camera Setup
def load_scene_from_xml(scene_filename="povo_scene.xml"):
  """Loads scene from XML file located relative to the project structure."""
  try:
    # Get the directory where the current script (sionnart.py) is located
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
  return Camera("def_camera", position=[22, 47, 110], look_at=[-18, 0, -8])

# FOR TESTING - Prints scene elements and associated materials
def print_scene_elements(scene):
  for i, obj in enumerate(scene.objects.values()):
    print(f"{obj.name} : {obj.radio_material.name}")
    if i >= 10:
      break

# Configures the antenna arrays for transmitters and receivers
def configure_antenna_arrays(scene):
  scene.tx_array = PlanarArray(
    num_rows=4, num_cols=4,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901", polarization="V"
  )

  scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="dipole", polarization="cross"
  )

# Set up and add a transmitter and receiver to the scene
def add_transmitter_receiver(scene, rx_pos, rx_ori):
  tx = Transmitter("tx", [-39, -15, 4])
  scene.add(tx)

  rx = Receiver("rx", rx_pos, rx_ori)
  scene.add(rx)

  tx.look_at(rx)  # Make the transmitter point at the receiver
  scene.frequency = 2.14e9  # Set frequency in Hz
  scene.synthetic_array = True  # Enable synthetic array for faster ray tracing

# Compute the propagation paths
def compute_paths(scene, max_depth=5, num_samples=1e6):
  paths = scene.compute_paths(max_depth=max_depth, num_samples=num_samples)
  return paths

# Finds the next available filename in the directory, with an incremented numeric suffix.
def get_next_filename(directory, base_name, extension):
  i = 1
  while True:
    filename = f"{base_name}_{i}.{extension}"
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
      return file_path
    i += 1

# Scene rendering and Saving
def render_and_save(scene, paths=None, camera="def_camera", no_preview=False, resolution=[480, 320], output_file="rendered_scene.png"):
  # Get the directory of the current script
  current_directory = os.path.dirname(os.path.abspath(__file__))
  # Create the "renders" directory if it doesn't exist
  renders_directory = os.path.join(current_directory, "renders")
  os.makedirs(renders_directory, exist_ok=True)

  # Automatically generate a unique filename with an incremental number
  output_path = get_next_filename(renders_directory, "paths_render", "png")
  
  if no_preview:
    print(f"Rendering without preview, saving to {output_path}")
    scene.render(camera=camera, paths=paths, show_devices=True, show_paths=True, num_samples=512, resolution=resolution)  
    plt.savefig(output_path, bbox_inches='tight')
  else:
    try:
      print("Attempting to render with preview...")
      scene.preview()
      scene.render(camera=camera, paths=paths, show_devices=True, show_paths=True, num_samples=512, resolution=resolution)
      plt.savefig(output_path, bbox_inches='tight')
    except RuntimeError as e:
      # If not available, fall back to regular rendering
      print(f"Preview not available, rendering with camera '{camera}' instead. Error: {e}")
      scene.render(camera=camera, num_samples=512, resolution=resolution)
      plt.savefig(output_path, bbox_inches='tight')
  
  print(f"Rendered image saved as {output_path}")

# Main execution
def main():

  # 1. Setup GPU and configure TensorFlow
  #setup_gpu()

  # 2. Parse arguments
  logger.info("Parsing command line arguments...")
  try:
    rx_pos, rx_ori = parse_arguments()
    logger.info(f"Received position: {rx_pos}, orientation: {rx_ori}")
  except Exception as e:
    logger.critical(f"Failed to parse arguments: {e}", exc_info=True)
    sys.exit(1)
  

  # 3. Check if preview mode is enabled
  no_preview = check_colab()

  # 4. Load scene and set camera
  logger.info("Loading scene...")
  scene = load_scene_from_xml("povo_scene.xml")
  if scene is None:
    logger.critical("Scene loading failed. Exiting.")
    sys.exit(1)

  logger.info("Setting up camera...")  
  scene.add(setup_camera())

  # 5. Configure antennas and add transmitter and dynamic receiver
  logger.info("Configuring antennas...")
  configure_antenna_arrays(scene)
  logger.info("Adding transmitter and receiver...")
  add_transmitter_receiver(scene, rx_pos, rx_ori)

  # 6. Compute paths and save renders
  logger.info("Computing paths...")
  paths = compute_paths(scene)
  logger.info("Rendering and saving scene...")
  render_and_save(scene, paths=paths, camera="def_camera", no_preview=no_preview, output_file="paths_render.png")

  logger.info("SionnaRT script finished successfully.")

if __name__ == "__main__":
  main()
