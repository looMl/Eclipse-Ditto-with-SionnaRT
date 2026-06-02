import trimesh
from pathlib import Path
from loguru import logger


def subdivide_mesh(input_path: Path, output_path: Path, max_edge_len: float = 1.0):
    """
    Loads a mesh, subdivides it so that no edge is longer than max_edge_len.
    """
    logger.info(f"Loading mesh for subdivision: {input_path}")
    try:
        mesh = trimesh.load(input_path, force="mesh")

        logger.debug(
            f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        # Iterative subdivision
        # Limit iterations to avoid explosion
        max_iter = 3

        for i in range(max_iter):
            edges = mesh.edges_unique_length
            max_len = edges.max()
            mean_len = edges.mean()
            logger.debug(
                f"Iteration {i}: max_edge_len={max_len:.2f}, mean_edge_len={mean_len:.2f}"
            )

            if max_len <= max_edge_len:
                logger.debug("Target edge length reached.")
                break

            # Subdivide
            new_vertices, new_faces = trimesh.remesh.subdivide(
                mesh.vertices, mesh.faces
            )
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        logger.debug(
            f"Subdivided mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        mesh.export(output_path)
        logger.success(f"Saved subdivided mesh to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to subdivide mesh: {e}")
        return False
