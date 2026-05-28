import pdal, json, sys, pathlib
import numpy as np
import trimesh

def tif_to_glb(input_file, output_file, sample_radius=10.0, max_edge=None, z_min=-500, z_max=3000):
    """
    Convert a GeoTIFF DEM to GLB for use in the Potree viewer.

    sample_radius : target point spacing in the DEM's native units (metres).
                    Larger = faster, fewer triangles, less detail.
                    For a 1 GB / 2 m TIF, start with 10 (→ ~10 m mesh).
    max_edge      : remove triangles with any edge longer than this (optional).
    z_min/z_max   : clamp elevation to exclude nodata fill values (e.g. -9999).
    """
    tmp = pathlib.Path(output_file).with_suffix(".tmp.glb")

    # Probe to find the actual band dimension name (varies by PDAL version)
    probe = pdal.Pipeline(json.dumps({
        "pipeline": [
            {"type": "readers.gdal", "filename": input_file},
            {"type": "filters.head", "count": 10}
        ]
    }))
    probe.execute()
    dims = probe.arrays[0].dtype.names
    print(f"Dimensions found: {dims}")
    band_dim = next((d for d in dims if d not in {"X", "Y", "Z"}), None)
    if band_dim is None:
        raise RuntimeError(f"Could not find a band dimension in: {dims}")
    print(f"Using band dimension: '{band_dim}'")

    pipeline = {
        "pipeline": [
            {"type": "readers.gdal", "filename": input_file},
            {"type": "filters.ferry", "dimensions": f"{band_dim}=>Z"},
            {"type": "filters.range", "limits": f"Z[{z_min}:{z_max}]"},
            {"type": "filters.sample", "radius": sample_radius},
            {"type": "filters.delaunay"},
            {"type": "writers.gltf", "filename": str(tmp)}
        ]
    }
    print(f"Running PDAL pipeline (sample_radius={sample_radius})…")
    pdal.Pipeline(json.dumps(pipeline)).execute()

    if max_edge is not None:
        print(f"Filtering faces with edge > {max_edge}…")
        scene = trimesh.load(str(tmp))
        mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
        v, f = mesh.vertices, mesh.faces
        e = np.stack([
            np.linalg.norm(v[f[:,1]] - v[f[:,0]], axis=1),
            np.linalg.norm(v[f[:,2]] - v[f[:,1]], axis=1),
            np.linalg.norm(v[f[:,0]] - v[f[:,2]], axis=1),
        ])
        mesh.update_faces(e.max(axis=0) < max_edge)
        mesh.export(output_file)
        tmp.unlink()
    else:
        tmp.rename(output_file)

    import os
    size_mb = os.path.getsize(output_file) / 1e6
    print(f"Done → {output_file}  ({size_mb:.1f} MB)")

# Usage:
#   python createDEMfromTIF.py input.tif output.glb [sample_radius] [max_edge] [z_min] [z_max]
#
# Iceland example (elevation 0–2200 m, start coarse):
#   python createDEMfromTIF.py iceland.tif iceland.glb 10 500
#
# Finer detail pass once coarse looks good:
#   python createDEMfromTIF.py iceland.tif iceland.glb 4 200

tif_to_glb(
    input_file    = sys.argv[1],
    output_file   = sys.argv[2],
    sample_radius = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0,
    max_edge      = float(sys.argv[4]) if len(sys.argv) > 4 else None,
    z_min         = float(sys.argv[5]) if len(sys.argv) > 5 else -500,
    z_max         = float(sys.argv[6]) if len(sys.argv) > 6 else 3000,
)
