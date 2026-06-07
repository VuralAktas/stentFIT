#!/usr/bin/env python
"""Stent skeletonisation of stent5.stl with Kimimaro (TEASAR).

Pipeline (only library functions, no hand-written skeleton method):
  1. trimesh.load(...)            -> read the original binary STL (unchanged)
  2. mesh.voxelized(pitch).fill() -> binary solid volume (trimesh)
  3. kimimaro.skeletonize(...)    -> TEASAR curve skeleton (vertices, edges, radius)
  4. vox.indices_to_points(...)   -> map skeleton vertices back to mm (trimesh)

Outputs CSV in the same layout as the in-house pipeline, plus a PNG.
"""
import os
import numpy as np
import trimesh
import kimimaro

HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "stent5.stl")
PITCH = 0.03   # mm per voxel (~3-4 voxels across a ~0.11 mm strut)

# --- 1. read original STL (binary, untouched) --------------------------------
mesh = trimesh.load(STL)
print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
      f"watertight={mesh.is_watertight}")

# --- 2. voxelise to a solid volume (trimesh) ---------------------------------
vox = mesh.voxelized(PITCH).fill()
vol = np.asarray(vox.matrix, dtype=np.uint8)   # label 1 = stent material
print(f"voxel volume: {vol.shape}  ({int(vol.sum())} filled voxels)")

# --- 3. Kimimaro TEASAR skeletonisation (package function) -------------------
skels = kimimaro.skeletonize(
    vol,
    teasar_params={
        "scale": 1.5,
        "const": 10,                       # voxel units
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500,
        "soma_detection_threshold": 750,
        "soma_invalidation_const": 300,
        "soma_invalidation_scale": 2,
    },
    anisotropy=(1, 1, 1),                  # work in voxel units, convert after
    dust_threshold=100,
    fix_branching=True,
    fix_borders=True,
    progress=False,
    parallel=1,
)
print(f"kimimaro returned {len(skels)} skeleton object(s): labels {list(skels.keys())}")
skel = max(skels.values(), key=lambda s: s.vertices.shape[0])

# --- 4. map skeleton vertices (voxel idx) back to mm (trimesh) ---------------
nodes = vox.indices_to_points(skel.vertices.astype(float))   # (N,3) mm
edges = np.asarray(skel.edges)                               # (E,2)
strut_radius = np.asarray(skel.radius) * PITCH               # mm (distance transform)
N = len(nodes)
print(f"skeleton: {N} nodes, {len(edges)} edges, "
      f"strut radius median {np.median(strut_radius):.4f} mm")

# graph connectivity
neighbors = [set() for _ in range(N)]
for a, b in edges:
    neighbors[int(a)].add(int(b)); neighbors[int(b)].add(int(a))
degree = np.array([len(s) for s in neighbors])
node_type = np.where(degree == 1, "endpoint",
             np.where(degree == 2, "line",
             np.where(degree == 0, "isolated", "junction")))

# cylindrical coords about the stent long axis
axis = int(np.argmax(nodes.max(0) - nodes.min(0)))
other = [i for i in range(3) if i != axis]
center = nodes[:, other].mean(0)
rel = nodes[:, other] - center
r = np.sqrt((rel ** 2).sum(1))
theta = np.arctan2(rel[:, 1], rel[:, 0])

# --- export ------------------------------------------------------------------
with open(os.path.join(HERE, "skeleton_points_kimimaro.csv"), "w") as f:
    f.write("skeleton_point_id,x,y,z,r,theta,strut_radius,node_type,degree,neighbor_ids\n")
    for i in range(N):
        nb = sorted(neighbors[i])
        f.write(f'{i},{nodes[i,0]},{nodes[i,1]},{nodes[i,2]},{r[i]},{theta[i]},'
                f'{strut_radius[i]},{node_type[i]},{degree[i]},"{nb}"\n')
with open(os.path.join(HERE, "skeleton_edges_kimimaro.csv"), "w") as f:
    f.write("edge_id,node_a,node_b\n")
    for k, (a, b) in enumerate(edges):
        f.write(f"{k},{int(a)},{int(b)}\n")
np.savez(os.path.join(HERE, "skeleton_kimimaro.npz"),
         nodes=nodes, edges=edges, strut_radius=strut_radius)

span = nodes[:, axis].max() - nodes[:, axis].min()
seg = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
summary = "\n".join([
    "Kimimaro (TEASAR) skeleton of stent5.stl",
    f"voxel pitch            : {PITCH} mm",
    f"skeleton nodes         : {N}",
    f"skeleton edges         : {len(edges)}",
    f"axial span             : {span:.3f} mm",
    f"strut radius median    : {np.median(strut_radius):.4f} mm",
    f"endpoints / junctions  : {(degree==1).sum()} / {(degree>=3).sum()}",
    f"segment len min/med/max: {seg.min():.4f} / {np.median(seg):.4f} / {seg.max():.4f} mm",
])
print(summary)
with open(os.path.join(HERE, "kimimaro_summary.txt"), "w") as f:
    f.write(summary + "\n")

# --- figure ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

segs = nodes[edges]
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(121, projection="3d")
order = [axis] + other
ax.add_collection3d(Line3DCollection(segs[:, :, order], colors="tab:red", lw=0.5))
ax.set_xlim(nodes[:, axis].min(), nodes[:, axis].max())
ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
ax.set_xlabel("axis (mm)"); ax.set_title(f"Kimimaro skeleton of stent5\n{N} nodes, {len(edges)} edges")
ax.view_init(elev=20, azim=-60)

ax2 = fig.add_subplot(122)
za = nodes[:, axis]
for a, b in edges:
    if abs(theta[b] - theta[a]) > np.pi:
        continue
    ax2.plot([theta[a], theta[b]], [za[a], za[b]], color="tab:red", lw=0.4)
ax2.set_xlabel("theta (rad)"); ax2.set_ylabel("axis (mm)"); ax2.set_title("Unrolled (theta vs axis)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "skeleton_kimimaro.png"), dpi=130)
print("wrote skeleton_points_kimimaro.csv, skeleton_edges_kimimaro.csv, skeleton_kimimaro.png")
