# GeoMap Processor

A robust Python module for generating high-fidelity 3D scenes for [SionnaRT](https://nvlabs.github.io/sionna/) simulations. This processor fetches geospatial data, processes Digital Elevation Models (DEM), generates building and infrastructure meshes, and orchestrates the creation of a scene compatible with ray-tracing engines.

> **Note:** This module leverages the [geo2sigmap](https://github.com/functions-lab/geo2sigmap) methodology for scene generation foundations.

## Overview

The `geomap_processor` is designed to automate the transition from real-world coordinates (Latitude/Longitude) to a simulation-ready 3D environment. It handles the complete pipeline: downloading terrain data, fetching OpenStreetMap (OSM) features, and normalizing geometry for accurate radio propagation simulation.

## Key Features

*   **Automated Terrain Processing**: Fetches 10m resolution DEM data (e.g., from TINItaly), reprojects it to WGS84, and aligns it with the scene origin.
*   **Building Meshing**: Generates and optimizes building meshes, ensuring they conform to the underlying terrain elevation to prevent "floating" or "buried" structures.
*   **Infrastructure Integration**: Fetches and meshes telecom infrastructure (antennas/towers) from OSM, applying vertical offsets based on local terrain height.

## Architecture

The module is organized into the following components:

```text
geomap_processor/
├── pipeline/
│   └── geo2sigmap.py       # Main orchestrator (SceneBuilder)
├── managers/
│   ├── building_manager.py # Handles building mesh merging and optimization
│   └── telecom_manager.py  # Fetches and meshes telecom infrastructure
├── processors/
│   └── dem_processor.py    # Core logic for DEM reprojection and normalization
├── data/
│   ├── dem_downloader.py   # WCS client for fetching DEM data
│   └── scene_updater.py    # Manages scene.xml I/O operations
└── utils/
    └── geometry_utils.py   # BoundingBox and coordinate utilities
```

### Workflow

1.  **Initialization**: `SceneBuilder` initializes with a target Bounding Box (BBox).
2.  **Core Generation**: The base scene (buildings, ground) is generated using external libraries.
3.  **Terrain Acquisition**: `DemDownloader` fetches the DEM for the area.
4.  **Terrain Processing**: `DemProcessor` generates a high-res terrain mesh and calculates the reference elevation at the scene center.
5.  **Height Adjustment**: A closure callback is created to map local scene coordinates to global coordinates and sample the DEM.
6.  **Optimization**: `BuildingManager` and `TelecomManager` apply this callback to vertically align all structures with the terrain.
7.  **Finalization**: The scene description (XML) is updated, and temporary artifacts are cleaned up.

## Usage

The module is typically invoked via the main application configuration:

```python
from app.geomap_processor.pipeline.geo2sigmap import SceneBuilder, BoundingBox, MaterialConfig

# Define area of interest
bbox = BoundingBox(
    min_lon=11.106, min_lat=46.056, 
    max_lon=11.153, max_lat=46.077
)

# Run generation
builder = SceneBuilder(output_dir=Path("./scene"))
builder.generate(bbox, MaterialConfig())
```

## Credits & References

*   **geo2sigmap**: This project builds upon the concepts and tools provided by the [geo2sigmap](https://github.com/functions-lab/geo2sigmap) repository.
*   **Sionna**: The generated scenes are optimized for NVIDIA's [Sionna](https://nvlabs.github.io/sionna/) library.
*   **TINITALY**: DEM data sourced from the [TINITALY](http://tinitaly.pi.ingv.it/) project.
*   **OpenStreetMap**: Building and infrastructure data provided by OSM contributors.
