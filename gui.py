"""
Simple GUI to select and run a FIB-SEM dataset through the pipeline.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from .pipeline import create_default_pipeline


def _run_pipeline_async(root: tk.Tk, path: Path, status_var: tk.StringVar):
    status_var.set("Running pipeline...")
    root.update_idletasks()
    try:
        p = create_default_pipeline()
        res = p.run_complete_pipeline(str(path))
        if res.get("error"):
            messagebox.showerror("Pipeline failed", res["error"])
            status_var.set("Failed")
        else:
            dur = res.get("pipeline_duration", 0.0)
            messagebox.showinfo("Done", f"Pipeline completed in {dur:.2f}s")
            status_var.set("Completed")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Error")


def launch_gui():
    root = tk.Tk()
    root.title("SEMSEG - FIB-SEM Pipeline")
    root.geometry("520x180")

    path_var = tk.StringVar()
    status_var = tk.StringVar(value="Idle")

    def browse():
        file = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("FIB-SEM data", ".tif .tiff .h5 .hdf5 .npy"),
                ("All files", "*.*"),
            ],
        )
        if file:
            path_var.set(file)

    def run_now():
        p = path_var.get().strip()
        if not p:
            messagebox.showwarning("Select file", "Please choose a data file.")
            return
        fp = Path(p)
        if not fp.exists():
            messagebox.showerror("Not found", f"File not found: {fp}")
            return
        threading.Thread(target=_run_pipeline_async, args=(root, fp, status_var), daemon=True).start()

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text="Data file:").grid(row=0, column=0, sticky="w")
    entry = tk.Entry(frm, textvariable=path_var, width=60)
    entry.grid(row=0, column=1, padx=6, sticky="we")
    tk.Button(frm, text="Browse...", command=browse).grid(row=0, column=2)

    tk.Button(frm, text="Run", command=run_now).grid(row=1, column=1, pady=12)
    tk.Label(frm, textvariable=status_var).grid(row=2, column=1)

    frm.columnconfigure(1, weight=1)
    root.mainloop()
