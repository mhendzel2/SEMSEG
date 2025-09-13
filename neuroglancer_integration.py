"""Neuroglancer integration helpers.

Optional dependency: python-neuroglancer
Install: pip install neuroglancer

Provides utilities to spin up a local Neuroglancer viewer with a Zarr / precomputed source
and capture ROI selections via annotation layers or crosshair position.
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any, Tuple

neuroglancer = None  # type: ignore
try:  # pragma: no cover - optional dependency
    import neuroglancer as _neuroglancer  # type: ignore
    neuroglancer = _neuroglancer  # type: ignore
    _HAS_NEUROGLANCER = True
except Exception:  # pragma: no cover
    _HAS_NEUROGLANCER = False

_viewer = None


def start_viewer(bind_address: str = '0.0.0.0', port: int = 0):
    """Start a Neuroglancer viewer server if not already started.
    Returns (viewer, url or None).
    """
    global _viewer
    if not _HAS_NEUROGLANCER:
        raise RuntimeError("python-neuroglancer not installed. pip install neuroglancer")
    if _viewer is not None:
        return _viewer, _viewer.state.url
    neuroglancer.set_server_bind_address(bind_address, port=port)  # type: ignore[union-attr]
    _viewer = neuroglancer.Viewer()  # type: ignore[union-attr]
    return _viewer, _viewer.state.url


def add_zarr_layer(viewer, name: str, zarr_path: str, voxel_size_nm=(8,8,8)):
    """Add a Zarr (or precomputed) layer to viewer.
    For Zarr s3 path: use fsspec + zarr to open and then wrap as local array if small.
    For large remote data, users should point to a precomputed source or directly use cloudvolume to generate one.
    """
    import numpy as np
    import fsspec, zarr  # type: ignore
    mapper = fsspec.get_mapper(zarr_path, anon=True)
    root = zarr.open(mapper, mode='r')
    arr = None
    if hasattr(root, 'shape'):
        arr = root
    else:
        for k, v in root.items():  # type: ignore[attr-defined]
            if hasattr(v, 'shape') and len(v.shape) >= 3:
                arr = v
                break
    if arr is None:
        raise RuntimeError("No array found in zarr root")
    # Warn about size
    if arr.size > 512**3:
        raise RuntimeError("Array too large to embed directly; provide precomputed source instead.")
    with viewer.txn() as s:  # type: ignore
    s.layers.append(name=name, layer=neuroglancer.LocalVolume(  # type: ignore[union-attr]
            data=arr,
            voxel_size=voxel_size_nm,
        ))


def add_remote_layer(viewer, name: str, source_url: str):
    """Add a remote source layer (e.g., precomputed://, zarr://) to the viewer."""
    if not _HAS_NEUROGLANCER:
        raise RuntimeError("python-neuroglancer not installed.")
    with viewer.txn() as s:  # type: ignore
        s.layers.append(name=name, layer=neuroglancer.ImageLayer(source=source_url))  # type: ignore[union-attr]


def get_first_image_layer_source(viewer) -> Optional[str]:
    """Return the source string of the first image layer in the viewer state, if any."""
    try:
        st = viewer.state  # type: ignore
        for lyr in st.layers:
            try:
                if getattr(lyr, 'layer_type', '') == 'image' or getattr(lyr, 'type', '') == 'image':
                    # python-neuroglancer uses .source on layer object
                    src = getattr(lyr, 'source', None)
                    if isinstance(src, str):
                        return src
            except Exception:
                continue
    except Exception:
        return None
    return None


def list_layers(viewer) -> Dict[str, Any]:
    """Return a summary of layers in the viewer state."""
    out: Dict[str, Any] = {"layers": []}
    try:
        st = viewer.state  # type: ignore
        for lyr in st.layers:
            try:
                out["layers"].append({
                    "name": getattr(lyr, 'name', None),
                    "type": getattr(lyr, 'layer_type', getattr(lyr, 'type', None)),
                    "source": getattr(lyr, 'source', None),
                })
            except Exception:
                continue
    except Exception:
        pass
    return out


def get_box_annotations(viewer, layer_name: Optional[str] = None) -> list:
    """Extract axis-aligned boxes from annotation layers.
    Returns list of dicts with 'corner1' and 'corner2' in (x,y,z) voxel coords.
    """
    boxes = []
    try:
        st = viewer.state  # type: ignore
        for lyr in st.layers:
            try:
                ltype = getattr(lyr, 'layer_type', getattr(lyr, 'type', ''))
                lname = getattr(lyr, 'name', '')
                if layer_name and lname != layer_name:
                    continue
                if ltype == 'annotation' and hasattr(lyr, 'annotations'):
                    for ann in getattr(lyr, 'annotations', []):
                        # python-neuroglancer uses dict-like annotations for boxes
                        if getattr(ann, 'type', getattr(ann, 'annotation_type', '')) in ('box', 'aabb') or 'pointA' in repr(ann):
                            # Try various attribute names
                            p1 = getattr(ann, 'pointA', getattr(ann, 'corner1', None))
                            p2 = getattr(ann, 'pointB', getattr(ann, 'corner2', None))
                            if p1 is not None and p2 is not None:
                                boxes.append({
                                    'layer': lname,
                                    'corner1': tuple(int(v) for v in list(p1)[:3]),
                                    'corner2': tuple(int(v) for v in list(p2)[:3]),
                                })
            except Exception:
                continue
    except Exception:
        pass
    return boxes


def get_crosshair_position(viewer) -> Tuple[int,int,int]:
    st = viewer.state  # type: ignore
    p = list(st.position)
    return int(p[0]), int(p[1]), int(p[2])


def build_roi(center_xyz: Tuple[int,int,int], size_xyz: Tuple[int,int,int]) -> Dict[str, slice]:
    cx, cy, cz = center_xyz
    sx, sy, sz = size_xyz
    return {
        'x': slice(max(cx - sx//2, 0), cx + sx//2),
        'y': slice(max(cy - sy//2, 0), cy + sy//2),
        'z': slice(max(cz - sz//2, 0), cz + sz//2),
    }
