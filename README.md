# Dynamic RF Scene Generation with SionnaRT

A toolkit that takes a geographic **bounding box** into a SionnaRT-ready 3D scene with
terrain, buildings, and a populated set of **cellular antennas**; it then runs
[NVIDIA SionnaRT](https://nvlabs.github.io/sionna/) GPU ray tracing on it to predict
radio coverage. The core of the project is the automated, reproducible path from
real-world coordinates to a simulation-ready environment, with no manual 3D modelling.

The reference scene is the historic centre of **Verona, Italy**, but the pipeline is
driven entirely by a bounding box in [`config.yaml`](src/config.yaml): point it at a
different area and it rebuilds the scene from scratch. 

> **Why this exists.** For realistic physical RF coverage simulation, one must have a 3D scene
> (terrain + buildings) *and* transmitter placement assumptions. Doing this by hand for
> a real area is tedious, introduces errors and is not efficient to "try another neighborhood."
> This toolkit automates that process: give it some coordinates, get a scene and antennas correctly setup for
> ray-tracing and pointing to the terrain, so that one can turn a map
> rectangle into an RF coverage prediction in minutes, enabling comparisons across actual geographies predictably.
> The goal is **not** a single painstakingly calibrated deployment
> but a fast, automated *starting point* for RF simulation. On the Verona
> reference scene, that automated baseline already lands within a centered RMSE (cRMSE) of
> **≈17 dB** of real walk-test RSRP without any calibration for that particular deployment which should provide a good base to improve on (see [Project status & scope](#project-status--scope)).

## Capabilities

- **Scene generation from a bounding box**: turn an arbitrary lat/lon rectangle into a
  SionnaRT-ready 3D scene: terrain from a DEM and buildings from OpenStreetMap, with no
  manual modelling.
- **Antenna population**: fetch telecom infrastructure from OpenStreetMap and emit a
  `transmitters.json` describing each site as a 3-sector cell (with electrical tilt),
  vertically aligned to the terrain.
- **Ray-traced coverage**: GPU path tracing for coverage maps / per-point RSRP
  predictions, with diffuse scattering and multi-bounce reflections.
- **Vegetation attenuation**: physics-based ITU-R P.833 (MED) excess loss applied along
  each ray path, driven by satellite + OSM vegetation data, *without* modifying the GPU scene.
- **Terminal-side effects**: hand-held body loss (ITU-T P.1238) and log-normal shadow fading.
- **3GPP TR 38.901 UMa baseline**: a closed-form Urban Macro path-loss model computed
  alongside the ray tracer for context.
- **Rendering**: visual scene renders and coverage overlays via Mitsuba.

## How it works

```
 bounding box                  scene.xml + meshes              coverage map
 (config.yaml)  ──────────►    + transmitters.json   ──────►   / per-point RSRP   ──────►  render
                geomap_processor          simulation            (+ vegetation loss)        / overlay
```

1. **`geomap_processor/`**: the geospatial → 3D scene pipeline, and the heart of the
   project. Downloads DEM terrain, fetches OSM buildings and telecom sites, aligns them to
   the terrain, and emits a Mitsuba/SionnaRT scene (`scene.xml` + meshes) plus a
   `transmitters.json` describing each antenna site and its sectors.
   See [`src/app/geomap_processor/README.md`](src/app/geomap_processor/README.md) for the
   internal architecture.
2. **`simulation/`**: the SionnaRT core: scene loading, ray-traced coverage maps, the
   ITU-R P.833 vegetation correction, terminal-side effects and
   Mitsuba rendering.

The two stages are decoupled: the scene-build step writes assets to disk, and the
simulation step reads them back, so you can rebuild a scene once and run many simulations
against it.

## Requirements

- **Python** 3.10 or 3.11 (`>=3.10, <3.12`).
- **NVIDIA GPU with CUDA** — SionnaRT ray tracing runs on the GPU.
- **LLVM 18.1.8** on your `PATH` — required by the Dr.Jit / Mitsuba backend that SionnaRT
  uses. On Windows, install `LLVM-18.1.8-win64.exe` from the
  [LLVM releases](https://github.com/llvm/llvm-project/releases?q=18.1.8&expanded=true).
- Python dependencies are declared in [`pyproject.toml`](pyproject.toml).

## Installation

This project uses [uv](https://docs.astral.sh/uv/). It resolves the `geo2sigmap` fork
declared under `[tool.uv.sources]` automatically and reproduces the locked environment in one
step:

```bash
git clone https://github.com/looMl/Eclipse-Ditto-with-SionnaRT.git
cd Eclipse-Ditto-with-SionnaRT
uv sync
```

## Quick start

The application package is `app`, rooted at `src/`, so run the modules from the `src/`
directory:

```bash
cd src

# 1. Build the 3D scene and populate antennas for the bounding box in config.yaml
uv run -m app.geomap_processor.pipeline.geo2sigmap

# 2a. Visual render of the scene
uv run -m app.simulation.simulator_cli render

# 2b. Coverage map (radio map + overlay). Add --no-vegetation for an ablation run.
uv run -m app.simulation.simulator_cli coverage
uv run -m app.simulation.simulator_cli coverage --no-vegetation
```

The scene is written to `src/scene/` (`scene.xml` + meshes) and the antenna definitions to
`src/ditto/things/transmitters.json`. Renders and coverage maps are written under
`src/renders/`.

## Configuration

Runtime settings live in [`src/config.yaml`](src/config.yaml), loaded and validated by
`src/app/config.py`. The main knobs:

| Key | What it controls |
| --- | --- |
| `geo2sigmap` | The `min_lon / min_lat / max_lon / max_lat` bounding box for scene generation, and the building `materials` (see [Changing building materials](#changing-building-materials)). |
| `sionnart.camera` | Camera pose (`position`, `orientation`, `look_at`) for renders. |
| `sionnart.rendering` | Render `resolution`, `num_samples` (quality vs. speed), `show_devices`. |
| `sionnart.coverage` | `samples_per_tx`, `max_depth`, `max_num_paths_per_src`, the `metric` (`rss` / `sinr` / `path_gain`), and the colour-scale `vmin` / `vmax`. |
| `sionnart.diffuse_scattering` | Scattering coefficient for diffuse (non-specular) energy. |
| `sionnart.handset.body_loss_db` | Hand-held body-shadowing loss (ITU-T P.1238). |
| `sionnart.shadowing` | Log-normal shadow-fading standard deviation (`sigma_db`). |
| `sionnart.vegetation` | ITU-R P.833 correction: `mode` (`per_link` / `per_path`), carrier `frequency_hz`, `leaf_state`, TCD/CHM data sources, raster step, and fallback canopy heights. |

> **GPU memory note.** Keep `samples_per_tx × num_tx ≤ 4,294,967,295` (uint32 limit), and
> size `max_num_paths_per_src` to your VRAM, the shipped default (`200000`) fits 16 GB at
> `max_depth=8` with diffuse scattering enabled. In our testing, coverage quality plateaus
> at `max_depth=8`: raising it further does not visibly change the result, it only costs
> more memory and time.

## Changing building materials

OSM gives building *geometry* but not construction materials, so the pipeline assigns an ITU
radio material to every surface at scene-build time. The three surfaces are set under
`geo2sigmap.materials` in [`src/config.yaml`](src/config.yaml), each an index into the ITU
material library (`scene_generation.itu_materials.ITU_MATERIALS`):

```yaml
geo2sigmap:
  materials:
    ground_idx: 1   # Concrete
    rooftop_idx: 2  # Brick
    wall_idx: 1     # Concrete
```

To use a different material, change the relevant index and re-run the scene-build step.
Available ITU materials and their valid frequency ranges:

| idx | Material | Frequency (GHz) |
| --: | --- | --- |
| 0 | Vacuum (≈air) | 0.001 – 100 |
| 1 | Concrete | 1 – 100 |
| 2 | Brick | 1 – 40 |
| 3 | Plasterboard | 1 – 100 |
| 4 | Wood | 0.001 – 100 |
| 5 | Glass | 0.1 – 100 / 220 – 450 |
| 6 | Ceiling Board | 1 – 100 / 220 – 450 |
| 7 | Chipboard | 1 – 100 |
| 8 | Plywood | 1 – 40 |
| 9 | Marble | 1 – 60 |
| 10 | Floorboard | 50 – 100 |
| 11 | Metal | 1 – 100 |
| 12 | Very Dry Ground | 1 – 10 |
| 13 | Medium Dry Ground | 1 – 10 |
| 14 | Wet Ground | 1 – 10 |
| 15 | Very Dry Ground (extended range) | 0.00001 – 300 |
| 16 | Medium Dry Ground (extended range) | 0.00001 – 300 |
| 17 | Wet Ground (extended range) | 0.00001 – 300 |

Properties follow ITU-R P.2040-2 (*Effects of building materials and structures on radiowave
propagation above about 100 MHz*); the extended-range ground variants (15–17) follow
ITU-R P.527-3.

For example, `wall_idx: 4` makes all building walls wood. Material changes only take effect
on the **next scene build**.

## Generating a scene for a different area

Change the `geo2sigmap` bounding box in [`src/config.yaml`](src/config.yaml) and re-run the
scene-build step.

Terrain is fetched from **TINITALY** (~10 m), which covers **Italy**, so any Italian
bounding box works out of the box. For regions **outside Italy**, only the *download* step
needs swapping: the downloader
([`dem_downloader.py`](src/app/geomap_processor/data/dem_downloader.py)) targets the TINITALY
WCS endpoint specifically, but `DemProcessor` (reprojection, meshing, terrain alignment) is
agnostic to where the DEM came from. Plug in your own DEM source (e.g. a Copernicus DEM API
for global coverage) that returns a GeoTIFF for the bounding box, and the rest of the pipeline runs unchanged.

## Optional: Eclipse Ditto integration

The scene-build step can optionally publish each generated antenna site as an
[Eclipse Ditto](https://www.eclipse.dev/ditto/) Thing. It is **off by default**; without the
flag the antennas are simply written to `src/ditto/things/transmitters.json` and consumed
locally by the simulation.

```bash
cd src
uv run -m app.geomap_processor.pipeline.geo2sigmap --ditto
```

This requires a running Ditto instance. A Docker Compose stack is provided under
[`docker/`](docker/) (see its [README](docker/README.md)). This path is **not needed** for
the scene-generation or simulation workflows above.

**Why it's here.** The integration was added with **digital twin** scenarios for smart
cities in mind. Representing each antenna as a Ditto Thing lets the simulated environment
stay synchronised with the real network: Things could carry live per-sector state
(configuration, traffic load, alarms) behind a uniform API, so a SionnaRT scene can be
re-evaluated whenever the network changes. That opens the door to closed-loop optimisation, e.g.
adjust a sector's tilt or power, re-run coverage to predict the effect, and push the chosen
configuration back.

## Repository layout

```
src/
├── config.yaml                 central runtime config (bounding box + simulation knobs)
├── scene/                      generated sionnart scene assets (scene.xml + .ply meshes)
├── ditto/things/               generated antenna definitions (transmitters.json)
├── renders/                    output: coverage maps + visual renders
└── app/
    ├── config.py               settings loader from yaml
    ├── geomap_processor/       bounding box → 3D scene + antenna population
    ├── simulation/             SionnaRT core
    │   ├── simulator_cli.py    entry point: render | coverage
    │   ├── coverage.py         radio-map computation
    │   ├── scene_manager.py    scene loading
    │   ├── vegetation/         ITU-R P.833 attenuation
    │   ├── baselines/          3GPP TR 38.901 UMa closed-form RSRP
    │   └── rendering/          visual + coverage rendering
    └── services/               Eclipse Ditto provisioning + RSRP readout helper
docker/                         optional Eclipse Ditto stack (Docker Compose)
pyproject.toml                  dependencies
```

## Project status & scope

This is a research project supporting a BSc thesis. It is **not** production software. The
scene-generation and simulation workflows are the supported entry points; the Eclipse Ditto
integration is optional and retained from the project's origins.

The aim is **not** a single, painstakingly calibrated area with verified per-sector antenna
configurations and surveyed surface materials. It is a tool to **quickly build a
more-than-decent scene with antennas already populated** for an arbitrary bounding box, a
fast, automated starting point for RF simulation. Antenna parameters (azimuth, tilt, power)
and material properties use reasonable defaults rather than ground-truth values, so results
are realistic at the scene level but not tuned to one specific deployment.

## Credits & references

- **[SionnaRT](https://nvlabs.github.io/sionna/)** — NVIDIA's GPU ray-tracing engine for RF
  propagation.
- **[geo2sigmap](https://github.com/functions-lab/geo2sigmap)** — scene-generation
  foundations used by `geomap_processor`.
- **[TINITALY](http://tinitaly.pi.ingv.it/)** — DEM terrain data (Italy).
- **[OpenStreetMap](https://www.openstreetmap.org/)** — building and telecom infrastructure data.
- **[ESA WorldCover](https://esa-worldcover.org/)** — tree-cover density for the vegetation model.
- **[Eclipse Ditto](https://www.eclipse.dev/ditto/)** — digital-twin platform (optional integration).
- **3GPP TR 38.901** — Urban Macro path-loss baseline.
- **ITU-R P.833** — vegetation attenuation model.
- **ITU-T P.1238** — hand-held body-shadowing loss.
- **ITU-R P.2040** — building-material electromagnetic properties (surface materials).
- **ITU-R P.527** — extended-range ground material properties.