"""
Streamlit Web UI for SEMSEG pipeline.

Launch with:
  streamlit run -q webui.py
or via CLI:
  python -m SEMSEG --web
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import io
import numpy as np
import streamlit as st  # type: ignore

# Optional dependencies
CloudVolume: Any | None = None
try:
    from cloudvolume import CloudVolume as _CloudVolume  # type: ignore
    CloudVolume = _CloudVolume
    _HAS_CLOUDVOLUME = True
except Exception:
    _HAS_CLOUDVOLUME = False

tifffile: Any | None = None
try:
    import tifffile as _tifffile  # type: ignore
    tifffile = _tifffile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False

h5py: Any | None = None
try:
    import h5py as _h5py  # type: ignore
    h5py = _h5py
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

label2rgb: Any | None = None
try:
    from skimage.color import label2rgb as _label2rgb  # type: ignore
    label2rgb = _label2rgb
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# Zarr/S3 support
zarr: Any | None = None
fsspec: Any | None = None
try:
    import zarr as _zarr  # type: ignore
    import fsspec as _fsspec  # type: ignore
    zarr = _zarr
    fsspec = _fsspec
    _HAS_ZARR = True
except Exception:
    _HAS_ZARR = False

# Plotly (optional) for 3D visualization
go: Any | None = None
try:
    import plotly.graph_objects as _go  # type: ignore
    go = _go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# Support running as a package module or as a standalone script
try:
    from .pipeline import create_default_pipeline  # type: ignore
except Exception:
    try:
        # Attempt absolute import when executed outside package context
        from SEMSEG.pipeline import create_default_pipeline  # type: ignore
    except Exception:
        # As a last resort, add parent of this file to sys.path
        pkg_root = Path(__file__).resolve().parent
        repo_root = pkg_root.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from SEMSEG.pipeline import create_default_pipeline  # type: ignore

try:
    from .core.config import FIBSEMConfig  # type: ignore
except Exception:
    try:
        from SEMSEG.core.config import FIBSEMConfig  # type: ignore
    except Exception:
        FIBSEMConfig = None  # type: ignore


st.set_page_config(page_title="SEMSEG Web UI", layout="wide")
st.title("SEMSEG – FIB-SEM Segmentation Pipeline")

# Session defaults
ss = st.session_state
ss.setdefault("input_path", "")
ss.setdefault("roi_bounds", {"z": [0, 0], "y": [0, 0], "x": [0, 0]})
ss.setdefault("roi_file", "")
ss.setdefault("preview_slice", {"axis": "z", "index": 0})
ss.setdefault("preproc", {"steps": ["noise_reduction", "contrast_enhancement"], "params": {}})
ss.setdefault("segment", {"method": "watershed", "type": "traditional"})
ss.setdefault("config", {})
ss.setdefault("config_path", None)
ss.setdefault("roi_queue", [])
ss.setdefault("remote_url", "")

tabs = st.tabs(["Data", "ROI", "Preprocess", "Segment", "Analyze", "3D Viewer", "Config"])

def _save_uploaded_file(buffer, name: str) -> Optional[Path]:
    if buffer is None:
        return None
    tmpdir = Path(tempfile.gettempdir())
    tmp = tmpdir / ("_upload_" + name)
    tmp.write_bytes(buffer.getbuffer())
    return tmp

def _preview_slice(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if axis == "z":
        index = int(np.clip(index, 0, arr.shape[0]-1))
        return arr[index]
    elif axis == "y":
        index = int(np.clip(index, 0, arr.shape[1]-1))
        return arr[:, index, :]
    else:
        index = int(np.clip(index, 0, arr.shape[2]-1))
        return arr[:, :, index]

def _load_preview_any(path: Path, axis: str, index: int) -> Optional[np.ndarray]:
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(path, mmap_mode="r")
            return _preview_slice(arr, axis, index)
        if ext in (".tif", ".tiff") and _HAS_TIFFFILE and tifffile is not None:
            with tifffile.TiffFile(path) as tif:  # type: ignore[union-attr]
                shp = tif.series[0].shape
                if len(shp) == 3:
                    # z,y,x -> read page index along first dim
                    index = int(np.clip(index, 0, shp[0]-1))
                    return tif.pages[index].asarray()
                else:
                    return tif.asarray()
        if ext in (".h5", ".hdf5") and _HAS_H5PY and h5py is not None:
            with h5py.File(path, "r") as f:  # type: ignore[union-attr]
                # choose the first dataset
                keys = list(f.keys())
                if not keys:
                    return None
                dset: Any = f[keys[0]]
                shp = dset.shape
                if len(shp) == 3:
                    if axis == "z":
                        index = int(np.clip(index, 0, shp[0]-1))
                        return dset[index, :, :]
                    if axis == "y":
                        index = int(np.clip(index, 0, shp[1]-1))
                        return dset[:, index, :]
                    index = int(np.clip(index, 0, shp[2]-1))
                    return dset[:, :, index]
                elif len(shp) == 2:
                    return dset[:]
        return None
    except Exception:
        return None

def _extract_roi_local(path: Path, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int) -> Optional[np.ndarray]:
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(path, mmap_mode="r")
            z1, y1, x1 = z0+dz, y0+dy, x0+dx
            return arr[z0:z1, y0:y1, x0:x1]
        if ext in (".tif", ".tiff") and _HAS_TIFFFILE and tifffile is not None:
            with tifffile.TiffFile(path) as tif:  # type: ignore[union-attr]
                shp = tif.series[0].shape
                if len(shp) != 3:
                    return None
                z1 = min(z0+dz, shp[0])
                slices = []
                for zi in range(z0, z1):
                    sl = tif.pages[zi].asarray()
                    slices.append(sl[y0:y0+dy, x0:x0+dx])
                if slices:
                    return np.stack(slices, axis=0)
                return None
        if ext in (".h5", ".hdf5") and _HAS_H5PY and h5py is not None:
            with h5py.File(path, "r") as f:  # type: ignore[union-attr]
                keys = list(f.keys())
                if not keys:
                    return None
                dset: Any = f[keys[0]]
                return dset[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        return None
    except Exception:
        return None

def _openorganelle_fetch(url: str, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int, mip: int = 0) -> np.ndarray:
    if not _HAS_CLOUDVOLUME:
        raise RuntimeError("cloudvolume is not installed. Install with: pip install cloud-volume")
    # Hint type checkers and guard runtime
    vol: Any = CloudVolume(url, mip=mip, progress=False, fill_missing=True)  # type: ignore[arg-type]
    # CloudVolume expects slices in zyx order with stop-exclusive indices
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    sub = vol[z0:z1, y0:y1, x0:x1]  # type: ignore[index]
    # Ensure numpy array, shape (z,y,x)
    arr = np.asarray(sub)
    if arr.ndim == 4:
        # Drop channels if present
        arr = arr[..., 0]
    return arr

def _zarr_fetch(url: str, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int, scale_key: str | None = None) -> np.ndarray:
    if not _HAS_ZARR:
        raise RuntimeError("zarr/fsspec not installed. Install with: pip install zarr fsspec s3fs")
    # Mapper supports s3://, gs://, http(s) via fsspec
    # Use anonymous access for public OpenOrganelle/COSEM buckets
    mapper = fsspec.get_mapper(url, anon=True)  # type: ignore[union-attr]
    root: Any = zarr.open(mapper, mode='r')  # type: ignore[union-attr]
    arr = None
    if hasattr(root, 'shape'):
        arr = root  # type: ignore[assignment]
    else:
        g = root  # type: ignore[assignment]
        # Try common multiscale keys
        if scale_key and scale_key in g:
            arr = g[scale_key]
        elif '0' in g:
            arr = g['0']
        elif 's0' in g:
            arr = g['s0']
        else:
            # Pick first 3D array
            for k, v in g.items():
                try:
                    if hasattr(v, 'shape') and len(v.shape) >= 3:  # type: ignore[attr-defined]
                        arr = v
                        break
                except Exception:
                    continue
    if arr is None:
        raise RuntimeError("No array found in Zarr store")
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    sub = arr[z0:z1, y0:y1, x0:x1]
    a = np.asarray(sub)
    if a.ndim == 4:
        a = a[..., 0]
    return a

def _zarr_fetch_lowest(url: str, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int) -> np.ndarray:
    if not _HAS_ZARR:
        raise RuntimeError("zarr/fsspec not installed. Install with: pip install zarr fsspec s3fs")
    mapper = fsspec.get_mapper(url, anon=True)  # type: ignore[union-attr]
    root: Any = zarr.open(mapper, mode='r')  # type: ignore[union-attr]
    if hasattr(root, 'shape'):
        arr = root  # type: ignore[assignment]
    else:
        g = root  # type: ignore[assignment]
        arrays = []
        for k, v in g.items():
            try:
                if hasattr(v, 'shape') and len(v.shape) >= 3:
                    arrays.append((k, v))
            except Exception:
                continue
        if not arrays:
            raise RuntimeError("No array found in Zarr store")
        arrays.sort(key=lambda kv: str(kv[0]))
        arr = arrays[-1][1]
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    sub = arr[z0:z1, y0:y1, x0:x1]
    a = np.asarray(sub)
    if a.ndim == 4:
        a = a[..., 0]
    return a

def _mip(arr: np.ndarray, axis: str) -> np.ndarray:
    if axis == 'z':
        return np.max(arr, axis=0)
    if axis == 'y':
        return np.max(arr, axis=1)
    return np.max(arr, axis=2)

def _maybe_decimate(arr: np.ndarray, max_dim: int = 256) -> np.ndarray:
    z, y, x = arr.shape[:3]
    sz = max(1, int(np.ceil(z / max_dim)))
    sy = max(1, int(np.ceil(y / max_dim)))
    sx = max(1, int(np.ceil(x / max_dim)))
    return arr[::sz, ::sy, ::sx]

def _plotly_volume(arr: np.ndarray, name: str = "volume"):
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly not installed. Install with: pip install plotly")
    a = _maybe_decimate(arr.astype(np.float32))
    a = (a - float(np.min(a))) / (float(np.max(a) - np.min(a)) + 1e-8)
    zz, yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]]
    fig = go.Figure(data=go.Volume(  # type: ignore[union-attr]
        x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
        value=a.flatten(),
        opacity=0.1,
        surface_count=12,
        colorscale='Gray'
    ))
    fig.update_layout(title=name, scene_aspectmode='data', margin=dict(l=0, r=0, t=30, b=0))
    return fig

def _plotly_isosurface(arr: np.ndarray, isovalue: float = 0.5, name: str = "isosurface"):
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly not installed. Install with: pip install plotly")
    a = _maybe_decimate(arr.astype(np.float32))
    zz, yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]]
    fig = go.Figure(data=go.Isosurface(  # type: ignore[union-attr]
        x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
        value=a.flatten(),
        isomin=isovalue, isomax=float(a.max()),
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale='Viridis',
        surface_count=2,
        opacity=0.9
    ))
    fig.update_layout(title=name, scene_aspectmode='data', margin=dict(l=0, r=0, t=30, b=0))
    return fig

# Data tab
with tabs[0]:
    st.subheader("Load Data")
    source = st.radio("Source", ["Local file", "OpenOrganelle (precomputed)"])
    local_col, remote_col = st.columns(2)
    with local_col:
        up = st.file_uploader("Upload (.tif/.h5/.hdf5/.npy)", type=["tif", "tiff", "h5", "hdf5", "npy"])
        p_text = st.text_input("Or path to file", ss["input_path"])
        if st.button("Use file"):
            if up is not None:
                path = _save_uploaded_file(up, up.name)
            else:
                path = Path(p_text) if p_text else None
            if path and Path(path).exists():
                ss["input_path"] = str(path)
                st.success(f"Selected: {path}")
            else:
                st.error("Please provide a valid file path or upload a file.")
    with remote_col:
        st.write("Enter an OpenOrganelle/Neuroglancer precomputed URL (e.g., gs:// or https URL)")
        url = st.text_input("Precomputed URL", ss.get("remote_url", ""))
        ss["remote_url"] = url
        mip = st.number_input("MIP Level (0=highest res)", min_value=0, max_value=10, value=0, step=1)
        if not _HAS_CLOUDVOLUME:
            st.info("cloudvolume not installed. Install with: pip install cloud-volume")
        st.caption("You can define ROI in the ROI tab and then fetch.")
        st.markdown("**Low-res remote preview**")
        prev_z0 = st.number_input("z start (preview)", min_value=0, value=0)
        prev_y0 = st.number_input("y start (preview)", min_value=0, value=0)
        prev_x0 = st.number_input("x start (preview)", min_value=0, value=0)
        prev_dz = st.number_input("depth dz (preview)", min_value=1, value=32)
        prev_dy = st.number_input("height dy (preview)", min_value=1, value=256)
        prev_dx = st.number_input("width dx (preview)", min_value=1, value=256)
        if st.button("Preview remote MIPs"):
            try:
                u = str(url or "")
                if u.endswith('.zarr'):
                    vol = _zarr_fetch_lowest(u, int(prev_z0), int(prev_y0), int(prev_x0), int(prev_dz), int(prev_dy), int(prev_dx))
                else:
                    vol = _openorganelle_fetch(u, int(prev_z0), int(prev_y0), int(prev_x0), int(prev_dz), int(prev_dy), int(prev_dx), int(mip))
                cmi1, cmi2, cmi3 = st.columns(3)
                with cmi1:
                    st.image(_mip(vol, 'z'), clamp=True, caption="MIP-Z")
                with cmi2:
                    st.image(_mip(vol, 'y'), clamp=True, caption="MIP-Y")
                with cmi3:
                    st.image(_mip(vol, 'x'), clamp=True, caption="MIP-X")
            except Exception as e:
                st.error(f"Remote preview failed: {e}")

# ROI tab
with tabs[1]:
    st.subheader("Region of Interest (ROI)")
    url = ss.get("remote_url", "")
    st.write("Define subregion to process. Provide bounds in voxels (z,y,x). Use Preview to see a slice.")
    z0 = st.number_input("z start", min_value=0, value=0)
    dz = st.number_input("depth (dz)", min_value=1, value=32)
    y0 = st.number_input("y start", min_value=0, value=0)
    dy = st.number_input("height (dy)", min_value=1, value=256)
    x0 = st.number_input("x start", min_value=0, value=0)
    dx = st.number_input("width (dx)", min_value=1, value=256)
    axis = st.selectbox("Preview axis", ["z", "y", "x"], index=0)
    idx = st.number_input("Preview index", min_value=0, value=0)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Fetch ROI from OpenOrganelle"):
            try:
                if not url:
                    st.error("Provide a precomputed URL in Data tab.")
                else:
                    arr = _openorganelle_fetch(url, int(z0), int(y0), int(x0), int(dz), int(dy), int(dx), int(mip))
                    # Save to temp NPY for pipeline
                    tmp = Path(tempfile.gettempdir()) / f"semseg_roi_{int(time.time())}.npy"
                    np.save(tmp, arr)
                    ss["roi_file"] = str(tmp)
                    ss["preview_slice"] = {"axis": axis, "index": int(idx)}
                    st.success(f"ROI saved: {tmp} shape={arr.shape}")
                    st.image(_preview_slice(arr, axis, int(idx)), clamp=True, caption="ROI preview")
            except Exception as e:
                st.error(f"Fetch failed: {e}")
        if st.button("Fetch ROI from Zarr S3"):
            try:
                if not url or not (url.startswith('s3://') and url.endswith('.zarr')):
                    st.error("Provide an s3://... .zarr URL in Data tab.")
                else:
                    arr = _zarr_fetch(url, int(z0), int(y0), int(x0), int(dz), int(dy), int(dx))
                    tmp = Path(tempfile.gettempdir()) / f"semseg_roi_{int(time.time())}.npy"
                    np.save(tmp, arr)
                    ss["roi_file"] = str(tmp)
                    ss["preview_slice"] = {"axis": axis, "index": int(idx)}
                    st.success(f"ROI saved: {tmp} shape={arr.shape}")
                    st.image(_preview_slice(arr, axis, int(idx)), clamp=True, caption="ROI preview")
            except Exception as e:
                st.error(f"Zarr fetch failed: {e}")
    with c2:
        if st.button("Preview local/selected file"):
            path = ss.get("roi_file") or ss.get("input_path")
            if not path or not Path(path).exists():
                st.error("No valid file selected.")
            else:
                sl = _load_preview_any(Path(path), axis, int(idx))
                if sl is None:
                    st.info("Preview supports NPY/TIFF/HDF5 (needs tifffile/h5py).")
                else:
                    st.image(sl, clamp=True, caption="Preview")

    st.divider()
    # Queue ROI for batch processing
    if st.button("Add ROI to Queue"):
        fmt = 'zarr' if (url and url.endswith('.zarr')) else 'precomputed'
        roi = {"z0": int(z0), "y0": int(y0), "x0": int(x0), "dz": int(dz), "dy": int(dy), "dx": int(dx), "mip": int(mip), "url": url, "format": fmt}
        ss["roi_queue"].append(roi)
        st.success(f"Queued ROI #{len(ss['roi_queue'])}")
    if ss["roi_queue"]:
        st.write("Queued ROIs:")
        st.json(ss["roi_queue"])
        if st.button("Clear ROI Queue"):
            ss["roi_queue"] = []

# Preprocess tab
with tabs[2]:
    st.subheader("Preprocessing Options")
    st.caption("These will be passed to the pipeline when running.")
    steps = st.multiselect("Steps", ["noise_reduction", "contrast_enhancement", "artifact_removal"], default=ss["preproc"]["steps"])
    nr_method = st.selectbox("Noise method", ["gaussian", "median", "bilateral", "wiener"], index=0)
    nr_sigma = st.number_input("Gaussian sigma", min_value=0.0, value=1.0)
    clahe_clip = st.number_input("CLAHE clip_limit", min_value=0.0, value=0.03)
    st.session_state["preproc"] = {
        "steps": steps,
        "params": {
            "noise_reduction": {"method": nr_method, "sigma": nr_sigma},
            "contrast_enhancement": {"clip_limit": clahe_clip},
        },
    }

# Segment tab
with tabs[3]:
    st.subheader("Segmentation")
    method = st.selectbox("Method", ["watershed", "thresholding", "morphology"], index=["watershed","thresholding","morphology"].index(ss["segment"]["method"]))
    seg_type = st.selectbox("Type", ["traditional", "deep_learning"], index=["traditional","deep_learning"].index(ss["segment"]["type"]))
    run_local = st.button("Run Pipeline on Selected (ROI if set, else file)")
    if run_local:
        path = ss.get("roi_file") or ss.get("input_path")
        if not path or not Path(path).exists():
            st.error("No valid file selected. Use Data/ROI tabs to select or fetch.")
        else:
            p = create_default_pipeline()
            params = {
                "preprocessing": {"steps": st.session_state["preproc"]["steps"], **st.session_state["preproc"]["params"]},
                "segmentation": {},
            }
            with st.spinner("Running pipeline..."):
                res = p.run_complete_pipeline(
                    path,
                    segmentation_method=method,
                    segmentation_type=seg_type,
                    **params,
                )
            if res.get("error"):
                st.error(f"Pipeline failed: {res['error']}")
            else:
                st.success(f"Done in {res.get('pipeline_duration', 0.0):.2f}s")
                st.json({k: v for k, v in res.items() if k in ("segmentation_method", "segmentation_type", "pipeline_duration")})
                # Save in session for Analyze tab
                st.session_state["_last_result"] = res

    # Batch process queued ROIs (OpenOrganelle or local NPY)
    if ss["roi_queue"] and st.button("Run Batch on ROI Queue"):
        outputs = []
        for i, roi in enumerate(ss["roi_queue"], start=1):
            try:
                # Prefer remote fetch if URL provided, else crop from selected file
                if roi.get("url"):
                    if roi.get("format") == 'zarr':
                        arr = _zarr_fetch(roi["url"], roi["z0"], roi["y0"], roi["x0"], roi["dz"], roi["dy"], roi["dx"])
                    else:
                        arr = _openorganelle_fetch(roi["url"], roi["z0"], roi["y0"], roi["x0"], roi["dz"], roi["dy"], roi["dx"], roi["mip"])
                else:
                    base_str = str(ss.get("input_path") or "")
                    base = Path(base_str) if base_str else None
                    if not base or not base.exists():
                        raise RuntimeError("No base file for local ROI.")
                    arr = _extract_roi_local(base, roi["z0"], roi["y0"], roi["x0"], roi["dz"], roi["dy"], roi["dx"])
                if arr is None:
                    raise RuntimeError("ROI extraction returned empty array")
                tmp = Path(tempfile.gettempdir()) / f"semseg_batch_{i}_{int(time.time())}.npy"
                np.save(tmp, arr)
                p = create_default_pipeline()
                res = p.run_complete_pipeline(tmp, segmentation_method=method, segmentation_type=seg_type)
                outputs.append({"roi": roi, "result": {k: res.get(k) for k in ("pipeline_duration",)}})
            except Exception as e:
                outputs.append({"roi": roi, "error": str(e)})
        st.json(outputs)

# Analyze tab
with tabs[4]:
    st.subheader("Analysis Results")
    res = st.session_state.get("_last_result")
    if not res:
        st.info("Run the pipeline first.")
    else:
        mh = res.get("morphological_quantification", {})
        pq = res.get("particle_quantification", {})
        st.write("Objects:", mh.get("morphological_analysis", {}).get("num_objects"))
        st.write("Particles:", pq.get("num_particles"))
        if mh:
            vols = mh.get("morphological_analysis", {}).get("volumes", [])
            if vols:
                st.bar_chart(np.array(vols))
        st.json({"morphology": mh, "particles": pq})

        # Segmentation overlay preview (if segmentation array is available)
        seg_block = res.get("segmentation_results", {})
        seg = seg_block.get("segmentation") if isinstance(seg_block, dict) else None
        src_path = st.session_state.get("roi_file") or st.session_state.get("input_path")
        if seg is not None and src_path and Path(src_path).exists():
            axis = st.selectbox("Overlay axis", ["z","y","x"], index=0)
            idx = st.number_input("Seg slice index", min_value=0, value=0)
            # Load preview of source to overlay
            img = _load_preview_any(Path(src_path), axis, int(idx))
            if img is not None:
                if _HAS_SKIMAGE and label2rgb is not None:
                    # Ensure labels and normalize background
                    try:
                        if seg.ndim == 3:
                            if axis == "z":
                                s = seg[min(int(idx), seg.shape[0]-1)]
                            elif axis == "y":
                                s = seg[:, min(int(idx), seg.shape[1]-1), :]
                            else:
                                s = seg[:, :, min(int(idx), seg.shape[2]-1)]
                        else:
                            s = seg
                        overlay = label2rgb(s.astype(int), image=img, bg_label=0, alpha=0.3)
                        st.image(overlay, caption="Segmentation Overlay")
                    except Exception:
                        st.image(img, caption="Source (overlay unavailable)")
                else:
                    st.image(img, caption="Source (install scikit-image for overlay)")

        # Downloads
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            try:
                import json as _json
                buf = io.BytesIO(_json.dumps(res, default=str, indent=2).encode("utf-8"))
                st.download_button("Download results JSON", buf, file_name="pipeline_results.json")
            except Exception:
                pass
        with cdl2:
            if seg is not None:
                buf2 = io.BytesIO()
                np.save(buf2, seg)
                buf2.seek(0)
                st.download_button("Download segmentation (.npy)", buf2, file_name="segmentation.npy")

# 3D Viewer tab
with tabs[5]:
    st.subheader("3D Viewer")
    st.caption("Interactive 3D volume and surface previews (decimated to fit in browser).")
    src = ss.get("roi_file") or ss.get("input_path")
    seg_res = st.session_state.get("_last_result", {}).get("segmentation_results", {}) if st.session_state.get("_last_result") else {}
    seg_arr = seg_res.get("segmentation") if isinstance(seg_res, dict) else None
    view_mode = st.selectbox("View", ["MIPs","Volume (intensity)", "Isosurface (segmentation)"])
    if view_mode == "MIPs":
        if src and Path(src).exists():
            arr = None
            try:
                if str(src).endswith('.npy'):
                    arr = np.load(src)
            except Exception:
                arr = None
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                cm1, cm2, cm3 = st.columns(3)
                with cm1: st.image(_mip(arr, 'z'), clamp=True, caption="MIP-Z")
                with cm2: st.image(_mip(arr, 'y'), clamp=True, caption="MIP-Y")
                with cm3: st.image(_mip(arr, 'x'), clamp=True, caption="MIP-X")
            else:
                st.info("Load an ROI (.npy) to view MIPs.")
        else:
            st.info("Select or fetch an ROI to view MIPs.")
    elif view_mode == "Volume (intensity)":
        if not _HAS_PLOTLY:
            st.info("Install plotly for 3D rendering: pip install plotly")
        elif src and Path(src).exists() and str(src).endswith('.npy'):
            try:
                vol = np.load(src)
                fig = _plotly_volume(vol, name="Intensity Volume")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Volume render failed: {e}")
        else:
            st.info("Volume rendering requires a loaded ROI (.npy). Use ROI tab to fetch.")
    else:
        if not _HAS_PLOTLY:
            st.info("Install plotly for 3D rendering: pip install plotly")
        elif seg_arr is None:
            st.info("Run segmentation first to view isosurface.")
        else:
            try:
                iso = (seg_arr > 0).astype(np.float32)
                iso_val = st.slider("Isovalue", min_value=0.1, max_value=1.0, value=0.5)
                fig = _plotly_isosurface(iso, isovalue=float(iso_val), name="Segmentation Surface")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Isosurface failed: {e}")

# Config tab
with tabs[5]:
    st.subheader("Configuration")
    st.caption("Advanced: edit configuration and apply when running the pipeline.")
    default_json = "{}"
    if FIBSEMConfig is not None:
        try:
            cfg = FIBSEMConfig()
            default_json = json.dumps(cfg.config, indent=2)
        except Exception:
            pass
    cfg_text = st.text_area("Configuration (JSON)", value=default_json, height=240)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Config to Temp"):
            try:
                data = json.loads(cfg_text)
                tmp = Path(tempfile.gettempdir()) / f"fibsem_config_{int(time.time())}.json"
                tmp.write_text(json.dumps(data, indent=2))
                ss["config_path"] = str(tmp)
                st.success(f"Saved: {tmp}")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
    with c2:
        if st.button("Run with Config (Selected Source)"):
            path = ss.get("roi_file") or ss.get("input_path")
            if not path or not Path(path).exists():
                st.error("No valid file selected.")
            else:
                p = create_default_pipeline()
                if ss.get("config_path"):
                    # Recreate pipeline with config
                    p = create_default_pipeline(config_path=ss["config_path"])
                res = p.run_complete_pipeline(path)
                if res.get("error"):
                    st.error(f"Failed: {res['error']}")
                else:
                    st.success(f"Done in {res.get('pipeline_duration', 0.0):.2f}s")
                    st.session_state["_last_result"] = res
