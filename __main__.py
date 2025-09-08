r"""
CLI entrypoint for SEMSEG.

Usage examples:
    python -m SEMSEG --diagnostics
    python -m SEMSEG --test
    python -m SEMSEG --run C:\\path\\to\\data.tif --method watershed --type traditional
    python -m SEMSEG --gui
    python -m SEMSEG --web
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import run_diagnostics, test_installation
from .pipeline import create_default_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="SEMSEG", description="FIB-SEM segmentation pipeline")
    parser.add_argument("--diagnostics", action="store_true", help="Run environment diagnostics")
    parser.add_argument("--test", action="store_true", help="Run a basic installation test")
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI file picker (Tkinter)")
    parser.add_argument("--web", action="store_true", help="Launch web UI (Streamlit)")
    parser.add_argument("--run", type=str, metavar="PATH", help="Run pipeline on a data file (.tif/.h5/.npy)")
    parser.add_argument("--method", type=str, default="watershed", help="Segmentation method (watershed|thresholding|morphology)")
    parser.add_argument("--type", dest="seg_type", type=str, default="traditional", help="Segmentation type (traditional|deep_learning)")
    parser.add_argument("--config", type=str, default=None, help="Optional config file path (json|yaml)")
    parser.add_argument("--output", type=str, default=None, help="Optional output directory for results")

    args = parser.parse_args(argv)

    if args.diagnostics:
        ok = run_diagnostics()
        return 0 if ok else 1

    if args.test:
        ok = test_installation()
        return 0 if ok else 1

    if args.gui:
        try:
            # Lazy import via importlib to avoid static analysis symbol errors
            mod = importlib.import_module(f"{__package__}.gui")
            launch_gui = getattr(mod, "launch_gui")
            launch_gui()
            return 0
        except Exception as e:
            print(f"GUI unavailable: {e}")
            return 1

    if args.web:
        try:
            # Attempt to launch the Streamlit app programmatically
            from streamlit.web import bootstrap  # type: ignore
            app_path = Path(__file__).with_name("webui.py")
            if not app_path.exists():
                print("Web UI app not found.")
                return 1
            bootstrap.run(str(app_path), args=[], flag_options={})
            return 0
        except ModuleNotFoundError:
            print("Streamlit is not installed. Install with: pip install streamlit")
            print("Then run: streamlit run -q \"" + str(Path(__file__).with_name("webui.py")) + "\"")
            return 1
        except Exception as e:
            print(f"Failed to launch web UI: {e}")
            return 1

    if args.run:
        data_path = Path(args.run)
        if not data_path.exists():
            print(f"Error: file not found: {data_path}")
            return 2

        pipeline = create_default_pipeline(config_path=args.config)
        try:
            results = pipeline.run_complete_pipeline(
                data_path,
                segmentation_method=args.method,
                segmentation_type=args.seg_type,
                output_dir=args.output,
            )
        except Exception as e:
            print(f"Pipeline error: {e}")
            return 1

        if 'error' in results:
            print(f"Pipeline failed: {results['error']}")
            return 1

        dur = results.get('pipeline_duration')
        print(f"Pipeline completed in {dur:.2f}s")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
