"""Throwaway: can free_transfer_point find anywhere at hand-off time?"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from apparecchiato.sim import layout as L

for seed in range(10):
    spec = L.sample_scene(seed)
    # World as it is when the plate hand-off runs under the rules plan: fork,
    # spoon and mug already in their slots, bottle where it spawned.
    world = {n: np.asarray(spec.goals[n], float) for n in ("fork", "spoon", "mug")}
    world["bottle"] = np.asarray(spec.by_name("bottle").pos, float)
    p = L.free_transfer_point(spec, world, "plate")
    base = np.asarray(spec.handoff_point, float)
    fell_back = bool(np.allclose(p[:2], base[:2]))
    scales = {o.kind: o.scale for o in spec.objects}
    near = min((float(np.linalg.norm(p[:2] - np.asarray(q)[:2]))
                - L.OBJECT_FOOTPRINT_R[k] * scales[k] - L.OBJECT_FOOTPRINT_R["plate"] * scales["plate"],
                k) for k, q in world.items())
    print(f"seed {seed}: transfer {np.round(p,4)} moved "
          f"{1000*np.linalg.norm(p[:2]-base[:2]):5.1f} mm  fallback={fell_back}  "
          f"gap={1000*near[0]:6.1f} mm to {near[1]}")
