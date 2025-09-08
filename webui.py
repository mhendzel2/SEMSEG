"""
Streamlit Web UI for SEMSEG pipeline.

Launch with:
  streamlit run -q webui.py
or via CLI:
  python -m SEMSEG --web
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st  # type: ignore

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


st.set_page_config(page_title="SEMSEG Web UI", layout="wide")
st.title("SEMSEG – FIB-SEM Segmentation Pipeline")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload data (.tif/.h5/.npy)", type=["tif", "tiff", "h5", "hdf5", "npy"])
    file_path_text = st.text_input("Or path to data file", "")

    st.header("Segmentation")
    method = st.selectbox("Method", ["watershed", "thresholding", "morphology"], index=0)
    seg_type = st.selectbox("Type", ["traditional", "deep_learning"], index=0)
    run_btn = st.button("Run Pipeline")

def _save_uploaded_file(buffer, name: str) -> Optional[Path]:
    if buffer is None:
        return None
    tmp = Path(st.session_state.get("_tmpdir", ".")) / ("_upload_" + name)
    tmp.write_bytes(buffer.getbuffer())
    return tmp

def _resolve_input_path() -> Optional[Path]:
    # Priority: uploaded file > text path
    if uploaded is not None:
        return _save_uploaded_file(uploaded, uploaded.name)
    p = file_path_text.strip()
    return Path(p) if p else None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run")
    if run_btn:
        path = _resolve_input_path()
        if path is None or not Path(path).exists():
            st.error("Please provide a valid input file via upload or path.")
        else:
            st.info(f"Running pipeline on: {path}")
            p = create_default_pipeline()
            start = time.time()
            res = p.run_complete_pipeline(
                path,
                segmentation_method=method,
                segmentation_type=seg_type,
            )
            dur = res.get("pipeline_duration", time.time() - start)
            if res.get("error"):
                st.error(f"Pipeline failed: {res['error']}")
            else:
                st.success(f"Completed in {dur:.2f}s")
                st.json({k: v for k, v in res.items() if k in ("segmentation_method", "segmentation_type", "pipeline_duration")})

with col2:
    st.subheader("Status")
    st.write("Ready" if not run_btn else "Done")
