import pdal, json, sys, pathlib
import numpy as np
import trimesh

def ground_mesh(input_file, output_file, radius=2.0, max_edge=60.0):
    tmp = pathlib.Path(output_file).with_suffix(".tmp.glb")

    pipeline = {
        "pipeline": [
            input_file,
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "filters.sample", "radius": radius},
            {"type": "filters.delaunay"},
            {"type": "writers.gltf", "filename": str(tmp)}
        ]
    }
    pdal.Pipeline(json.dumps(pipeline)).execute()

    # Remove faces with any edge longer than max_edge (same effect as max_edge_length)
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
    print(f"Exported {len(mesh.faces)} faces → {output_file}")

# python createDEMmesh.py input.copc.laz ground.glb [sample_radius] [max_edge_length]
input_f   = sys.argv[1]
output_f  = sys.argv[2]
r         = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
max_edge  = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

ground_mesh(input_f, output_f, radius=r, max_edge=max_edge)
