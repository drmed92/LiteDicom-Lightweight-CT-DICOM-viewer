#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteDicom - a lightweight DICOM viewer / voxel inspector
=========================================================
Single-file Tkinter viewer for CT / MR DICOM series with HU windowing,
colormaps, HU->material auto-coloring, orthogonal (XY/XZ/YZ) views,
voxel probing, measuring (line + polygon area), pencil / highlighter /
eraser tools, zoom (slider + scroll-wheel) and mouse pan.

Requirements (Windows 10, Python 3.8+):
    pip install pydicom numpy pillow
    # optional, only for JPEG/RLE-compressed DICOM:
    #   pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm

Run:
    python dicom_viewer.py

A concept by Ahmed I.  (ahmedalj@pm.me / back292@pm.me)
"""

import os
import sys
import math
import queue
import random
import colorsys
import threading
import subprocess
import importlib.util


# ---- prerequisite auto-install ----------------------------------------------
def _ensure_packages(packages):
    """packages: list of (import_name, pip_name).  Auto-install any missing ones."""
    # When frozen into an .exe (PyInstaller), all dependencies are bundled and
    # there is no pip to call, so skip the check entirely.
    if getattr(sys, "frozen", False):
        return
    missing = [(imp, pip) for imp, pip in packages
               if importlib.util.find_spec(imp) is None]
    if not missing:
        return
    print("=" * 66)
    print(" LiteDicom: the following prerequisites are missing and will now be")
    print(" installed automatically:  " + ", ".join(pip for _, pip in missing))
    print("=" * 66)
    for imp, pip in missing:
        print(" ->  pip install %s" % pip)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip])
        except Exception as e:
            sys.exit("\n Could not auto-install '%s' (%s)."
                     "\n Please install it manually:  pip install %s\n" % (pip, e, pip))
    print("\n All prerequisites installed.  Starting LiteDicom ...\n")


# numpy / pillow / pydicom can be installed via pip; do that first if needed.
_ensure_packages([("numpy", "numpy"), ("PIL", "pillow"), ("pydicom", "pydicom")])

import numpy as np                       # noqa: E402
import pydicom                           # noqa: E402
from PIL import Image                    # noqa: E402

# Tkinter ships with the standard Windows/macOS Python installer and CANNOT be
# pip-installed, so if it is somehow absent we guide the user rather than crash.
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, colorchooser
    from PIL import ImageTk
except Exception as e:
    sys.exit(
        "\n ERROR: Tkinter is not available in this Python installation.\n"
        " Tkinter is part of the standard library but was not bundled here.\n"
        "   Windows / macOS : reinstall Python from python.org "
        "(Tcl/Tk is included by default).\n"
        "   Linux           : install it, e.g.  sudo apt install python3-tk\n"
        "\n (details: %s)\n" % e)


# =============================================================================
#  Constants / defaults
# =============================================================================

# Window presets: name -> (level, width)  in HU
PRESETS = {
    "Soft tissue": (40, 400),
    "Lung":        (-600, 1500),
    "Bone":        (300, 1500),
    "Brain":       (40, 80),
    "Mediastinum": (50, 350),
    "Abdomen":     (50, 350),
    "Liver":       (30, 150),
    "Full range":  (0, 2000),
}

# Default HU -> material table (contiguous, Schneider-flavoured, editable in-app).
# Ranges are [low, high).  Colors are assigned from the active palette by order.
DEFAULT_MATERIALS = [
    {"name": "Air",              "low": -1024, "high": -950},
    {"name": "Lung",             "low": -950,  "high": -500},
    {"name": "Adipose / Fat",    "low": -500,  "high": -50},
    {"name": "Water / CSF",      "low": -50,   "high": 15},
    {"name": "Soft tissue",      "low": 15,    "high": 120},
    {"name": "Trabecular bone",  "low": 120,   "high": 500},
    {"name": "Cortical bone",    "low": 500,   "high": 1500},
    {"name": "Metal / Dense",    "low": 1500,  "high": 32767},
]

# Continuous colormap anchor points (pos 0..1 -> RGB)
_CMAP_ANCHORS = {
    "Grayscale": [(0, (0, 0, 0)), (1, (255, 255, 255))],
    "Bone":      [(0, (0, 0, 0)), (0.375, (83, 83, 116)),
                  (0.75, (166, 199, 199)), (1, (255, 255, 255))],
    "Hot":       [(0, (0, 0, 0)), (0.365, (255, 0, 0)),
                  (0.746, (255, 255, 0)), (1, (255, 255, 255))],
    "Jet":       [(0, (0, 0, 131)), (0.125, (0, 60, 170)), (0.375, (5, 255, 255)),
                  (0.625, (255, 255, 0)), (0.875, (250, 0, 0)), (1, (128, 0, 0))],
    "Cool":      [(0, (0, 255, 255)), (1, (255, 0, 255))],
    "Rainbow":   [(0, (128, 0, 255)), (0.25, (0, 0, 255)), (0.5, (0, 255, 0)),
                  (0.75, (255, 255, 0)), (1, (255, 0, 0))],
    "Viridis":   [(0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                  (0.75, (94, 201, 98)), (1, (253, 231, 37))],
    "Inferno":   [(0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
                  (0.75, (249, 142, 9)), (1, (252, 255, 164))],
}

# Discrete palettes for material auto-coloring (cycled if more materials)
_PALETTES = {
    "Vivid":  [(50, 50, 60), (80, 160, 255), (255, 210, 120), (90, 220, 120),
               (255, 90, 90), (200, 120, 255), (255, 160, 40), (240, 240, 240)],
    "Pastel": [(70, 70, 80), (150, 190, 235), (245, 220, 170), (170, 220, 170),
               (240, 160, 160), (200, 175, 225), (235, 200, 150), (230, 230, 235)],
    "Earth":  [(60, 55, 50), (120, 150, 170), (200, 175, 120), (110, 160, 90),
               (170, 90, 70), (150, 120, 90), (210, 180, 130), (235, 230, 220)],
    "Neon":   [(30, 30, 40), (0, 200, 255), (255, 240, 0), (0, 255, 120),
               (255, 0, 150), (180, 0, 255), (255, 120, 0), (255, 255, 255)],
    "Warm":   [(45, 40, 45), (120, 90, 160), (240, 200, 120), (220, 150, 90),
               (230, 90, 70), (200, 70, 110), (250, 180, 90), (245, 235, 220)],
    "Cool":   [(40, 45, 55), (70, 130, 200), (120, 200, 220), (90, 200, 160),
               (150, 170, 230), (110, 140, 220), (170, 210, 235), (235, 240, 245)],
}


# =============================================================================
#  Module-level helpers (kept GUI-free so they can be unit-tested)
# =============================================================================

def build_lut(anchors):
    """Return a 256x3 uint8 lookup table from (pos, rgb) anchor points."""
    xs = np.linspace(0.0, 1.0, 256)
    pos = [a[0] for a in anchors]
    lut = np.zeros((256, 3), np.uint8)
    for ch in range(3):
        vals = [a[1][ch] for a in anchors]
        lut[:, ch] = np.clip(np.interp(xs, pos, vals), 0, 255).astype(np.uint8)
    return lut


def get_luts():
    return {name: build_lut(anchors) for name, anchors in _CMAP_ANCHORS.items()}


def material_palette(name, n):
    """Return an (n,3) uint8 palette, cycling the named base palette."""
    base = _PALETTES.get(name, _PALETTES["Vivid"])
    out = np.zeros((n, 3), np.uint8)
    for i in range(n):
        out[i] = base[i % len(base)]
    return out


def classify_indices(hu, materials):
    """Vectorised HU -> material index (int array, same shape as hu)."""
    idx = np.zeros(hu.shape, np.int32)
    for i, m in enumerate(materials):
        mask = (hu >= m["low"]) & (hu < m["high"])
        idx[mask] = i
    return idx


def classify_name(value, materials):
    for m in materials:
        if m["low"] <= value < m["high"]:
            return m["name"]
    return "Unknown"


def polygon_area(pts):
    """Shoelace area of a polygon given as [(x,y), ...] in pixel units."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _first_val(v, default=None):
    """pydicom values can be MultiValue; grab the first sensible scalar."""
    if v is None:
        return default
    try:
        if isinstance(v, (list, tuple)) or hasattr(v, "__len__") and not isinstance(v, str):
            return v[0]
    except Exception:
        pass
    return v


def load_dicom_series(paths, progress=None):
    """
    Load one or more DICOM files into a 3-D HU volume.

    `paths` may be a folder (walked recursively) or a list of file paths.
    `progress` is an optional callback(str) used for live log messages.
    Non-DICOM / corrupt / undecodable files are skipped and reported; the
    function never raises on a per-file problem.

    Returns (volume[z,y,x] int16, spacing=(sz,sy,sx), meta dict, n_loaded, n_skipped),
    or (None, None, None, 0, n_skipped) if nothing usable was found.
    """
    log = progress if progress else (lambda *_: None)

    files = []
    if isinstance(paths, str) and os.path.isdir(paths):
        for root, _, fs in os.walk(paths):
            for f in fs:
                files.append(os.path.join(root, f))
    elif isinstance(paths, (list, tuple)):
        files = list(paths)
    else:
        files = [paths]

    log("Scanning %d file(s)..." % len(files))
    datasets = []
    skipped = 0
    for p in files:
        base = os.path.basename(p)
        try:
            ds = pydicom.dcmread(p, force=True)
        except Exception:
            log("  failed to load DCM file: %s" % base)
            skipped += 1
            continue
        if "PixelData" not in ds or not hasattr(ds, "Rows"):
            log("  ignored (not a DICOM image): %s" % base)
            skipped += 1
            continue
        datasets.append(ds)

    if not datasets:
        log("No valid DICOM image files were found.")
        return None, None, None, 0, skipped

    # --- spatial sort ---
    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is not None and iop is not None:
            try:
                r = np.array(iop[:3], float)
                c = np.array(iop[3:6], float)
                nrm = np.cross(r, c)
                return float(np.dot(np.array(ipp, float), nrm))
            except Exception:
                pass
        return float(getattr(ds, "InstanceNumber", 0) or 0)

    datasets.sort(key=sort_key)
    ds0 = datasets[0]
    rows, cols = int(ds0.Rows), int(ds0.Columns)

    # --- in-plane spacing ---
    ps = getattr(ds0, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        sy, sx = float(ps[0]), float(ps[1])
    else:
        sy = sx = 1.0

    # --- slice spacing from positions, else header ---
    zpos = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        zpos.append(float(ipp[2]) if ipp is not None else None)
    if len([z for z in zpos if z is not None]) > 1:
        zs = np.array([z for z in zpos if z is not None], float)
        diffs = np.abs(np.diff(np.sort(zs)))
        diffs = diffs[diffs > 1e-4]
        sz = float(np.median(diffs)) if len(diffs) else 1.0
    else:
        sz = float(getattr(ds0, "SpacingBetweenSlices", None)
                   or getattr(ds0, "SliceThickness", None) or 1.0)
    if sz <= 0:
        sz = 1.0

    # --- build volume (decode here so we can report progress + skip failures) ---
    total = len(datasets)
    hus, kept = [], []
    for i, ds in enumerate(datasets):
        if i % 5 == 0 or i == total - 1:
            log("Loading slices %d/%d..." % (i + 1, total))
        base = os.path.basename(getattr(ds, "filename", "") or ("slice %d" % i))
        try:
            arr = ds.pixel_array
        except Exception:
            log("  failed to decode (compressed? install pylibjpeg/gdcm): %s" % base)
            skipped += 1
            continue
        if arr.shape != (rows, cols):
            log("  skipped mismatched slice: %s" % base)
            skipped += 1
            continue
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        hu = arr.astype(np.float32) * slope + inter
        hus.append(np.clip(np.round(hu), -32768, 32767).astype(np.int16))
        kept.append(ds)

    if not hus:
        log("No slices could be decoded.")
        return None, None, None, 0, skipped

    vol = np.stack(hus, axis=0)
    n_ok = len(hus)
    ds0 = kept[0]
    log("Done: %d slice(s) loaded, %d skipped." % (n_ok, skipped))

    # --- metadata (radiologist-relevant fields) ---
    def g(attr, default="-"):
        v = getattr(ds0, attr, None)
        return default if v in (None, "") else v

    meta = {
        "PatientID":        g("PatientID"),
        "PatientSex":       g("PatientSex"),
        "PatientAge":       g("PatientAge"),
        "Modality":         g("Modality"),
        "StudyDescription": g("StudyDescription"),
        "SeriesDescription":g("SeriesDescription"),
        "StudyDate":        g("StudyDate"),
        "AcquisitionTime":  g("AcquisitionTime", g("SeriesTime")),
        "Institution":      g("InstitutionName"),
        "Manufacturer":     g("Manufacturer"),
        "BodyPart":         g("BodyPartExamined"),
        "PatientPosition":  g("PatientPosition"),
        "KVP":              g("KVP"),
        "TubeCurrent_mA":   g("XRayTubeCurrent"),
        "Exposure_mAs":     g("Exposure"),
        "ExposureTime_ms":  g("ExposureTime"),
        "CTDIvol":          g("CTDIvol"),
        "Kernel":           _first_val(getattr(ds0, "ConvolutionKernel", None), "-"),
        "GantryTilt":       g("GantryDetectorTilt"),
        "Contrast":         g("ContrastBolusAgent"),
        "SliceThickness":   g("SliceThickness"),
        "Photometric":      g("PhotometricInterpretation"),
        "BitsStored":       g("BitsStored"),
        "WinCenter":        _first_val(getattr(ds0, "WindowCenter", None), None),
        "WinWidth":         _first_val(getattr(ds0, "WindowWidth", None), None),
        "Rows":             rows,
        "Cols":             cols,
        "Slices":           n_ok,
        "PixelSpacing":     (sy, sx),
        "SliceSpacing":     sz,
    }
    return vol, (sz, sy, sx), meta, n_ok, skipped


# =============================================================================
#  The viewer
# =============================================================================

class DicomViewer:
    AXES = ("Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)")

    def __init__(self, root):
        self.root = root
        root.title("LiteDicom  -  lightweight DICOM viewer")
        # size to the actual screen so the layout fits on small displays
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(1280, int(sw * 0.95))
        h = min(820, int(sh * 0.90))
        root.geometry("%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2), max(0, (sh - h) // 3)))
        root.minsize(min(900, w), min(520, h))
        self._compact = sh < 820           # tighten padding on short screens

        # ---- state ----
        self.volume = None                 # (z,y,x) int16 HU
        self.spacing = (1.0, 1.0, 1.0)     # (sz, sy, sx)
        self.meta = {}
        self.axis = self.AXES[0]
        self.idx = 0
        self.wl, self.ww = 40.0, 400.0
        self.zoom = 1.0
        self.pan_x, self.pan_y = 40, 40
        self.luts = get_luts()
        self.materials = [dict(m) for m in DEFAULT_MATERIALS]

        self.cur_slice = None              # 2-D int16 (oriented + flipped)
        self.ar = 1.0                      # pixel aspect (row_mm / col_mm)
        self.col_mm = 1.0
        self.row_mm = 1.0
        self.base_img = None               # aspect-corrected coloured PIL image
        self.base_w = self.base_h = 1
        self.tk_img = None
        self._base_dirty = True

        # annotations keyed by (axis, idx) -> list of stroke dicts
        self.annotations = {}
        self.temp = None                   # in-progress drawing state
        self._poly_deck = []               # shuffled colour deck for polygons

        # ---- tk variables ----
        self.var_axis = tk.StringVar(value=self.axis)
        self.var_cmap = tk.StringVar(value="Grayscale")
        self.var_palette = tk.StringVar(value="Vivid")
        self.var_autocolor = tk.BooleanVar(value=False)
        self.var_flip_h = tk.BooleanVar(value=False)
        self.var_flip_v = tk.BooleanVar(value=False)
        self.var_tool = tk.StringVar(value="pan")
        self.var_pencolor = tk.StringVar(value="#ff3030")
        self.var_penwidth = tk.IntVar(value=3)
        self.var_verbose = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Open a DICOM folder or file to begin.")
        self.var_x = tk.StringVar(value="0")
        self.var_y = tk.StringVar(value="0")
        self.var_z = tk.StringVar(value="0")

        self.marker = None                 # (axis, idx, col, row) crosshair
        self._loading = False              # guard against overlapping loads
        self._load_queue = None

        self._build_ui()
        self._bind_canvas()

    # -------------------------------------------------------------- UI build
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # top bar ----------------------------------------------------------
        top = ttk.Frame(self.root, padding=(6, 4))
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open folder…", command=self.open_folder).pack(side="left")
        ttk.Button(top, text="Open file…", command=self.open_file).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="↻ Update view", command=self.force_refresh).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(top, text="Verbose log", variable=self.var_verbose).pack(side="right")

        # bottom status bar (packed first so it stays lowest) --------------
        status = ttk.Frame(self.root)
        status.pack(side="bottom", fill="x")
        ttk.Label(status, textvariable=self.var_status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(fill="x")

        # full-width slice slider along the bottom of the screen -----------
        slicebar = ttk.Frame(self.root, padding=(8, 3))
        slicebar.pack(side="bottom", fill="x")
        ttk.Label(slicebar, text="Slice").pack(side="left")
        self.slice_scale = tk.Scale(slicebar, from_=0, to=0, orient="horizontal",
                                    command=self.on_slice, showvalue=False)
        self.slice_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.slice_lbl = ttk.Label(slicebar, text="- / -", width=12)
        self.slice_lbl.pack(side="left")

        # main split -------------------------------------------------------
        main = ttk.Frame(self.root)
        main.pack(side="top", fill="both", expand=True)

        # left control column (scrollable via scrollbar OR mouse wheel)
        left_outer = ttk.Frame(main, width=270)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)
        lc = tk.Canvas(left_outer, width=270, highlightthickness=0, bg="#f0f0f0")
        lsb = ttk.Scrollbar(left_outer, orient="vertical", command=lc.yview)
        self.ctrl = ttk.Frame(lc)
        self.ctrl.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        self._lc_window = lc.create_window((0, 0), window=self.ctrl, anchor="nw")
        # keep inner frame the full width of the canvas (no horizontal gap/scroll)
        lc.bind("<Configure>", lambda e: lc.itemconfigure(self._lc_window, width=e.width))
        lc.configure(yscrollcommand=lsb.set)
        lc.pack(side="left", fill="both", expand=True)
        lsb.pack(side="right", fill="y")
        self._build_controls(self.ctrl)
        self._enable_wheel_scroll(lc, self.ctrl)

        # center: canvas + floating info box + log
        center = ttk.Frame(main)
        center.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(center, bg="#101014", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="top", fill="both", expand=True)

        # floating DICOM info box (bottom-left of canvas)
        self.info_lbl = tk.Label(center, justify="left", anchor="nw",
                                 bg="#0b0b0f", fg="#8fe6b0",
                                 font=("Consolas", 8), bd=1, relief="solid", padx=6, pady=4)
        self.info_lbl.place(in_=self.canvas, relx=0.0, rely=1.0, x=8, y=-8, anchor="sw")
        self.info_lbl.configure(text="No DICOM loaded.")

        # log panel
        logf = ttk.Frame(center)
        logf.pack(side="bottom", fill="x")
        ttk.Label(logf, text="Log", padding=(4, 0)).pack(side="top", anchor="w")
        self.log_txt = tk.Text(logf, height=5, wrap="word", font=("Consolas", 8),
                               bg="#1c1c22", fg="#c8c8d0", bd=0)
        self.log_txt.pack(side="left", fill="x", expand=True)
        lsb2 = ttk.Scrollbar(logf, orient="vertical", command=self.log_txt.yview)
        lsb2.pack(side="right", fill="y")
        self.log_txt.configure(yscrollcommand=lsb2.set, state="disabled")

    def _section(self, parent, title):
        pad = (8, 4) if self._compact else (8, 6)
        lf = ttk.LabelFrame(parent, text=title, padding=pad)
        lf.pack(fill="x", padx=6, pady=(3 if self._compact else 6, 0))
        return lf

    def _enable_wheel_scroll(self, canvas, container):
        """Scroll the left control panel with the mouse wheel over any of its
        widgets, without disturbing the image canvas's own wheel handling."""
        def _on(e):
            num = getattr(e, "num", None)
            if num == 4:
                step = -1
            elif num == 5:
                step = 1
            else:
                step = -1 if e.delta > 0 else 1
            canvas.yview_scroll(step, "units")
            return "break"

        def bind_tree(w):
            w.bind("<MouseWheel>", _on)
            w.bind("<Button-4>", _on)
            w.bind("<Button-5>", _on)
            for ch in w.winfo_children():
                bind_tree(ch)

        canvas.bind("<MouseWheel>", _on)
        canvas.bind("<Button-4>", _on)
        canvas.bind("<Button-5>", _on)
        bind_tree(container)

    def _build_controls(self, p):
        # view ------------------------------------------------------------
        s = self._section(p, "View")
        om = ttk.OptionMenu(s, self.var_axis, self.axis, *self.AXES,
                            command=lambda _=None: self.on_axis_change())
        om.pack(fill="x")
        row = ttk.Frame(s); row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(row, text="Flip H", variable=self.var_flip_h,
                        command=self.force_refresh).pack(side="left")
        ttk.Checkbutton(row, text="Flip V", variable=self.var_flip_v,
                        command=self.force_refresh).pack(side="left", padx=(8, 0))

        # go to voxel -----------------------------------------------------
        s = self._section(p, "Go to voxel")
        gr = ttk.Frame(s); gr.pack(fill="x")
        for i, (lab, var) in enumerate((("X", self.var_x), ("Y", self.var_y), ("Z", self.var_z))):
            ttk.Label(gr, text=lab).grid(row=0, column=i * 2, padx=(0 if i == 0 else 4, 1))
            e = ttk.Entry(gr, textvariable=var, width=5)
            e.grid(row=0, column=i * 2 + 1)
            e.bind("<Return>", lambda ev: self.jump_to())
        self.coord_range = ttk.Label(s, text="", foreground="#666")
        self.coord_range.pack(anchor="w", pady=(3, 0))
        ttk.Button(s, text="Go / show point", command=self.jump_to).pack(fill="x", pady=(3, 0))

        # windowing -------------------------------------------------------
        s = self._section(p, "Windowing (HU)")
        ttk.Label(s, text="Level").pack(anchor="w")
        self.wl_scale = tk.Scale(s, from_=-1200, to=3000, orient="horizontal",
                                 command=self.on_wl, showvalue=True)
        self.wl_scale.set(self.wl); self.wl_scale.pack(fill="x")
        ttk.Label(s, text="Width").pack(anchor="w")
        self.ww_scale = tk.Scale(s, from_=1, to=4000, orient="horizontal",
                                 command=self.on_ww, showvalue=True)
        self.ww_scale.set(self.ww); self.ww_scale.pack(fill="x")
        pf = ttk.Frame(s); pf.pack(fill="x", pady=(4, 0))
        for i, name in enumerate(PRESETS):
            b = ttk.Button(pf, text=name, width=11,
                           command=lambda n=name: self.apply_preset(n))
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=1, pady=1)
        pf.columnconfigure(0, weight=1); pf.columnconfigure(1, weight=1)

        # colour ----------------------------------------------------------
        s = self._section(p, "Colouring")
        ttk.Label(s, text="Colormap (HU)").pack(anchor="w")
        ttk.OptionMenu(s, self.var_cmap, self.var_cmap.get(), *self.luts.keys(),
                       command=lambda _=None: self.force_refresh()).pack(fill="x")
        ttk.Checkbutton(s, text="Auto-colour by material", variable=self.var_autocolor,
                        command=self.force_refresh).pack(anchor="w", pady=(6, 0))
        ttk.Label(s, text="Material palette").pack(anchor="w")
        ttk.OptionMenu(s, self.var_palette, self.var_palette.get(), *_PALETTES.keys(),
                       command=lambda _=None: self.force_refresh()).pack(fill="x")
        ttk.Button(s, text="Edit material table…",
                   command=self.open_material_editor).pack(fill="x", pady=(4, 0))

        # zoom ------------------------------------------------------------
        s = self._section(p, "Zoom")
        self.zoom_scale = tk.Scale(s, from_=10, to=2000, orient="horizontal",
                                   command=self.on_zoom_slider, showvalue=True)
        self.zoom_scale.set(100); self.zoom_scale.pack(fill="x")
        zf = ttk.Frame(s); zf.pack(fill="x")
        ttk.Button(zf, text="Fit", command=self.fit_view).pack(side="left", expand=True, fill="x")
        ttk.Button(zf, text="1:1", command=lambda: self.set_zoom(1.0)).pack(side="left", expand=True, fill="x")

        # tools -----------------------------------------------------------
        s = self._section(p, "Tools")
        tools = [("Pan", "pan"), ("Window drag", "window"), ("Probe HU", "probe"),
                 ("Measure line", "line"), ("Measure area", "poly"),
                 ("Pencil", "pencil"), ("Highlighter", "hl"),
                 ("Eraser", "erase"), ("Magic eraser", "magic")]
        for label, val in tools:
            ttk.Radiobutton(s, text=label, value=val, variable=self.var_tool,
                            command=self.on_tool_change).pack(anchor="w")

        # pen -------------------------------------------------------------
        s = self._section(p, "Pen / highlight")
        cf = ttk.Frame(s); cf.pack(fill="x")
        swatches = ["#000000", "#ffffff", "#ff3030", "#30d030", "#3060ff",
                    "#ffd000", "#00d0d0", "#ff30ff", "#ff9000"]
        for i, c in enumerate(swatches):
            b = tk.Button(cf, bg=c, width=2, relief="raised",
                          command=lambda c=c: self.set_pen_color(c))
            b.grid(row=i // 5, column=i % 5, padx=1, pady=1)
        self.pen_prev = tk.Label(s, text="  selected  ", bg=self.var_pencolor.get(),
                                 fg="#ffffff")
        self.pen_prev.pack(fill="x", pady=(4, 0))
        ttk.Button(s, text="Custom colour…", command=self.pick_pen_color).pack(fill="x", pady=(2, 0))
        ttk.Label(s, text="Width").pack(anchor="w")
        tk.Scale(s, from_=1, to=20, orient="horizontal", variable=self.var_penwidth,
                 showvalue=True).pack(fill="x")
        ttk.Button(s, text="Clear this slice", command=self.clear_slice).pack(fill="x", pady=(6, 0))

        ttk.Label(p, text="a concept by Ahmed I.", foreground="#888",
                  padding=(6, 8)).pack(anchor="w")

    # -------------------------------------------------------------- logging
    def log(self, msg, verbose=False):
        if verbose and not self.var_verbose.get():
            return
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", msg + "\n")
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")

    # -------------------------------------------------------------- loading
    def open_folder(self):
        d = filedialog.askdirectory(title="Select a DICOM series folder")
        if d:
            self._load(d)

    def open_file(self):
        f = filedialog.askopenfilenames(
            title="Select DICOM file(s)",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")])
        if f:
            self._load(list(f))

    def _load(self, paths):
        """Kick off loading on a background thread so the UI never freezes."""
        if self._loading:
            self.log("A load is already in progress; please wait.")
            return
        self._loading = True
        self.var_status.set("Loading…")
        self.log("── Loading ──")
        self._load_queue = queue.Queue()
        t = threading.Thread(target=self._load_worker, args=(paths,), daemon=True)
        t.start()
        self.root.after(40, self._poll_load)

    def _load_worker(self, paths):
        """Runs off-thread: only touches the queue, never Tk widgets."""
        def prog(msg):
            self._load_queue.put(("log", msg))
        try:
            result = load_dicom_series(paths, progress=prog)
            self._load_queue.put(("done", result))
        except Exception as e:                       # unexpected only
            self._load_queue.put(("error", str(e)))

    def _poll_load(self):
        """Runs on the main thread: drains log messages, applies the result."""
        try:
            while True:
                kind, payload = self._load_queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "error":
                    self._loading = False
                    self.var_status.set("Load failed.")
                    self.log("LOAD ERROR: " + payload)
                    return
                elif kind == "done":
                    self._loading = False
                    self._finalize_load(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(40, self._poll_load)

    def _finalize_load(self, result):
        vol, spacing, meta, n_ok, skipped = result
        if vol is None:
            self.var_status.set("Nothing loaded (%d file(s) skipped)." % skipped)
            return

        self.volume = vol
        self.spacing = spacing
        self.meta = meta
        self.annotations.clear()
        self._poly_deck = []
        self.marker = None

        self.log("Volume %s, spacing (z,y,x)=%.3f, %.3f, %.3f mm"
                 % (vol.shape, spacing[0], spacing[1], spacing[2]))

        # initial window from header if present, else default soft-tissue
        wc, wwid = meta.get("WinCenter"), meta.get("WinWidth")
        try:
            if wc is not None and wwid is not None:
                self.wl, self.ww = float(wc), float(wwid)
            else:
                self.wl, self.ww = 40.0, 400.0
        except Exception:
            self.wl, self.ww = 40.0, 400.0
        self.wl_scale.set(self.wl); self.ww_scale.set(self.ww)

        z, y, x = vol.shape
        self.coord_range.configure(text="range: X 0-%d  Y 0-%d  Z 0-%d" % (x - 1, y - 1, z - 1))
        self.axis = self.var_axis.get()
        self.on_axis_change(initial=True)
        self.root.after(60, self.fit_view)
        self.update_info_panel()
        self.var_status.set("Loaded %s  |  HU range %d … %d"
                            % (str(vol.shape), int(vol.min()), int(vol.max())))

    # -------------------------------------------------------------- axis / slice
    def _axis_len(self):
        z, y, x = self.volume.shape
        a = self.var_axis.get()
        if a == self.AXES[0]:
            return z
        if a == self.AXES[1]:
            return y
        return x

    def on_axis_change(self, initial=False):
        if self.volume is None:
            return
        self.axis = self.var_axis.get()
        n = self._axis_len()
        mid = n // 2
        self.slice_scale.configure(from_=0, to=max(0, n - 1))
        if initial or self.idx >= n:
            self.idx = mid
        self.slice_scale.set(self.idx)
        self.extract_slice()
        self._base_dirty = True
        self.fit_view()

    def on_slice(self, val):
        if self.volume is None:
            return
        self.idx = int(float(val))
        self.slice_lbl.configure(text="%d / %d" % (self.idx + 1, self._axis_len()))
        self.extract_slice()
        self._base_dirty = True
        self.redraw()
        self.update_info_panel()
        self.log("slice -> %d" % self.idx, verbose=True)

    def extract_slice(self):
        """Pull the current 2-D HU slice, orient it, set spacing/aspect."""
        z, y, x = self.volume.shape
        sz, sy, sx = self.spacing
        a = self.var_axis.get()
        i = min(self.idx, self._axis_len() - 1)

        if a == self.AXES[0]:          # Axial XY : rows=y cols=x
            sl = self.volume[i, :, :]
            self.col_mm, self.row_mm = sx, sy
        elif a == self.AXES[1]:        # Coronal XZ : rows=z cols=x
            sl = self.volume[:, i, :]
            sl = np.flipud(sl)         # head up
            self.col_mm, self.row_mm = sx, sz
        else:                          # Sagittal YZ : rows=z cols=y
            sl = self.volume[:, :, i]
            sl = np.flipud(sl)         # head up
            self.col_mm, self.row_mm = sy, sz

        if self.var_flip_h.get():
            sl = np.fliplr(sl)
        if self.var_flip_v.get():
            sl = np.flipud(sl)

        self.cur_slice = np.ascontiguousarray(sl)
        self.ar = (self.row_mm / self.col_mm) if self.col_mm else 1.0
        self.slice_lbl.configure(text="%d / %d" % (i + 1, self._axis_len()))

    # -------------------------------------------------------------- windowing
    def on_wl(self, val):
        self.wl = float(val); self._base_dirty = True; self.redraw(); self.draw_overlay()

    def on_ww(self, val):
        self.ww = max(1.0, float(val)); self._base_dirty = True; self.redraw(); self.draw_overlay()

    def apply_preset(self, name):
        wl, ww = PRESETS[name]
        self.wl, self.ww = float(wl), float(ww)
        self.wl_scale.set(wl); self.ww_scale.set(ww)
        self._base_dirty = True
        self.redraw()
        self.log("preset %s  (WL %g / WW %g)" % (name, wl, ww))

    # -------------------------------------------------------------- colouring
    def colorize(self):
        arr = self.cur_slice.astype(np.float32)
        lo, hi = self.wl - self.ww / 2.0, self.wl + self.ww / 2.0
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        if self.var_autocolor.get():
            idx = classify_indices(arr, self.materials)
            pal = material_palette(self.var_palette.get(), len(self.materials))
            rgb = pal[idx].astype(np.float32)
            rgb *= (0.35 + 0.65 * norm[..., None])       # keep anatomy visible
            rgb = rgb.clip(0, 255).astype(np.uint8)
        else:
            u8 = (norm * 255).astype(np.uint8)
            rgb = self.luts[self.var_cmap.get()][u8]

        img = Image.fromarray(rgb, "RGB")
        H, W = rgb.shape[0], rgb.shape[1]
        dh = max(1, int(round(H * self.ar)))
        if dh != H:
            img = img.resize((W, dh), Image.BILINEAR)
        self.base_img = img
        self.base_w, self.base_h = W, dh

    # -------------------------------------------------------------- transforms
    def img_to_canvas(self, cx, ry):
        return (self.pan_x + cx * self.zoom,
                self.pan_y + ry * self.ar * self.zoom)

    def canvas_to_img(self, x, y):
        return ((x - self.pan_x) / self.zoom,
                (y - self.pan_y) / (self.ar * self.zoom))

    # -------------------------------------------------------------- rendering
    def redraw(self):
        if self.volume is None or self.cur_slice is None:
            return
        if self._base_dirty or self.base_img is None:
            self.colorize()
            self._base_dirty = False

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        z = self.zoom

        bx0 = int(max(0, math.floor((-self.pan_x) / z)))
        by0 = int(max(0, math.floor((-self.pan_y) / z)))
        bx1 = int(min(self.base_w, math.ceil((cw - self.pan_x) / z)))
        by1 = int(min(self.base_h, math.ceil((ch - self.pan_y) / z)))

        self.canvas.delete("img")
        if bx1 > bx0 and by1 > by0:
            crop = self.base_img.crop((bx0, by0, bx1, by1))
            dw = max(1, int(round((bx1 - bx0) * z)))
            dh = max(1, int(round((by1 - by0) * z)))
            resample = Image.NEAREST if z >= 2 else Image.BILINEAR
            disp = crop.resize((dw, dh), resample)
            self.tk_img = ImageTk.PhotoImage(disp)
            self.canvas.create_image(self.pan_x + bx0 * z, self.pan_y + by0 * z,
                                     anchor="nw", image=self.tk_img, tags="img")
        self.canvas.tag_lower("img")
        self.draw_annotations()
        self.draw_overlay()

    def draw_overlay(self):
        self.canvas.delete("ovl")
        if self.volume is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        a = self.var_axis.get()
        # orientation letters (approximate radiological convention)
        if a == self.AXES[0]:
            lt, rt, tp, bt = "R", "L", "A", "P"
        elif a == self.AXES[1]:
            lt, rt, tp, bt = "R", "L", "S", "I"
        else:
            lt, rt, tp, bt = "A", "P", "S", "I"
        if self.var_flip_h.get():
            lt, rt = rt, lt
        if self.var_flip_v.get():
            tp, bt = bt, tp
        col = "#a8f0c0"
        f = ("Consolas", 10, "bold")
        self.canvas.create_text(10, ch // 2, text=lt, fill=col, font=f, anchor="w", tags="ovl")
        self.canvas.create_text(cw - 10, ch // 2, text=rt, fill=col, font=f, anchor="e", tags="ovl")
        self.canvas.create_text(cw // 2, 8, text=tp, fill=col, font=f, anchor="n", tags="ovl")
        self.canvas.create_text(cw // 2, ch - 8, text=bt, fill=col, font=f, anchor="s", tags="ovl")
        # top-right technique line
        info = "WL %g  WW %g   Zoom %d%%   Slice %d/%d" % (
            self.wl, self.ww, int(self.zoom * 100), self.idx + 1, self._axis_len())
        self.canvas.create_text(cw - 10, 8, text=info, fill="#cfcfd8",
                                font=("Consolas", 9), anchor="ne", tags="ovl")

        # jump-to-voxel crosshair marker (only on its own axis + slice)
        if self.marker and self.marker[0] == a and self.marker[1] == self.idx:
            mcol, mrow = self.marker[2], self.marker[3]
            mx, my = self.img_to_canvas(mcol + 0.5, mrow + 0.5)
            r = 12
            self.canvas.create_line(mx - r, my, mx + r, my, fill="#ff2fa0", width=1, tags="ovl")
            self.canvas.create_line(mx, my - r, mx, my + r, fill="#ff2fa0", width=1, tags="ovl")
            self.canvas.create_oval(mx - 4, my - 4, mx + 4, my + 4, outline="#ff2fa0",
                                    width=1, tags="ovl")

    # -------------------------------------------------------------- annotations
    def _key(self):
        return (self.var_axis.get(), self.idx)

    def _list(self):
        return self.annotations.setdefault(self._key(), [])

    def draw_annotations(self):
        self.canvas.delete("anno")
        for st in self._list():
            self._draw_stroke(st)
        if self.temp:
            self._draw_temp()

    def _cpts(self, pts):
        out = []
        for cx, ry in pts:
            x, y = self.img_to_canvas(cx, ry)
            out.extend([x, y])
        return out

    def _draw_stroke(self, st):
        t = st["t"]
        if t == "pencil":
            if len(st["pts"]) >= 2:
                self.canvas.create_line(*self._cpts(st["pts"]), fill=st["color"],
                                        width=st["w"], capstyle="round", joinstyle="round",
                                        smooth=True, tags="anno")
        elif t == "hl":
            if len(st["pts"]) >= 2:
                self.canvas.create_line(*self._cpts(st["pts"]), fill=st["color"],
                                        width=st["w"] * 3, capstyle="round", joinstyle="round",
                                        stipple="gray50", smooth=True, tags="anno")
        elif t == "line":
            p = st["p"]
            x0, y0 = self.img_to_canvas(*p[0]); x1, y1 = self.img_to_canvas(*p[1])
            self.canvas.create_line(x0, y0, x1, y1, fill="#00ffd0", width=2,
                                    arrow="both", tags="anno")
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            self.canvas.create_text(mx, my - 8, text=st["label"], fill="#00ffd0",
                                    font=("Consolas", 9, "bold"), tags="anno")
        elif t == "poly":
            coords = self._cpts(st["v"])
            self.canvas.create_polygon(*coords, fill=st["color"], stipple="gray50",
                                       outline=st["color"], width=2, tags="anno")
            cxs = st["v"]
            cx = sum(p[0] for p in cxs) / len(cxs)
            ry = sum(p[1] for p in cxs) / len(cxs)
            tx, ty = self.img_to_canvas(cx, ry)
            self.canvas.create_text(tx, ty, text=st["label"], fill="#ffffff",
                                    font=("Consolas", 9, "bold"), tags="anno")

    def _draw_temp(self):
        t = self.temp["t"]
        if t in ("pencil", "hl"):
            pts = self.temp["pts"]
            if len(pts) >= 2:
                w = self.var_penwidth.get() * (3 if t == "hl" else 1)
                kw = dict(fill=self.var_pencolor.get(), width=w, capstyle="round",
                          joinstyle="round", smooth=True, tags="anno")
                if t == "hl":
                    kw["stipple"] = "gray50"
                self.canvas.create_line(*self._cpts(pts), **kw)
        elif t == "line":
            p0 = self.temp["p0"]; p1 = self.temp.get("p1")
            if p1:
                x0, y0 = self.img_to_canvas(*p0); x1, y1 = self.img_to_canvas(*p1)
                self.canvas.create_line(x0, y0, x1, y1, fill="#00ffd0", width=2,
                                        arrow="both", dash=(4, 3), tags="anno")
        elif t == "poly":
            vs = self.temp["v"]
            cs = self._cpts(vs)
            if len(vs) >= 2:
                self.canvas.create_line(*cs, fill="#ffd000", width=2, tags="anno")
            cur = self.temp.get("cur")
            if cur and vs:
                x0, y0 = self.img_to_canvas(*vs[-1]); x1, y1 = self.img_to_canvas(*cur)
                self.canvas.create_line(x0, y0, x1, y1, fill="#ffd000", width=1,
                                        dash=(3, 3), tags="anno")
            for cx, ry in vs:
                x, y = self.img_to_canvas(cx, ry)
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffd000",
                                        outline="", tags="anno")

    # -------------------------------------------------------------- poly colours
    def _next_poly_color(self):
        if not self._poly_deck:
            hues = [i / 14.0 for i in range(14)]
            random.shuffle(hues)
            deck = []
            for h in hues:
                r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
                deck.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
            self._poly_deck = deck
        return self._poly_deck.pop()

    # -------------------------------------------------------------- measurements
    def _line_label(self, p0, p1):
        dx = (p1[0] - p0[0]) * self.col_mm
        dy = (p1[1] - p0[1]) * self.row_mm
        d = math.hypot(dx, dy)
        if self.col_mm != 1.0 or self.row_mm != 1.0:
            return "%.1f mm" % d
        return "%.0f px" % math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    def _poly_label(self, vs):
        area_px = polygon_area(vs)
        if self.col_mm != 1.0 or self.row_mm != 1.0:
            return "%.1f mm²" % (area_px * self.col_mm * self.row_mm)
        return "%.0f px²" % area_px

    # -------------------------------------------------------------- zoom / pan
    def set_zoom(self, z, mx=None, ch=None):
        if self.volume is None:
            return
        z = max(0.1, min(20.0, z))
        if mx is None:
            cw = self.canvas.winfo_width(); ch2 = self.canvas.winfo_height()
            mx, ch = cw / 2, ch2 / 2
        # keep the image point under the cursor fixed
        cx, ry = self.canvas_to_img(mx, ch)
        self.zoom = z
        self.pan_x = mx - cx * self.zoom
        self.pan_y = ch - ry * self.ar * self.zoom
        self.zoom_scale.set(int(z * 100))
        self.redraw()

    def on_zoom_slider(self, val):
        if self.volume is None:
            return
        self.set_zoom(float(val) / 100.0)

    def fit_view(self):
        if self.volume is None:
            return
        if self._base_dirty or self.base_img is None:
            self.colorize(); self._base_dirty = False
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        z = min(cw / self.base_w, ch / self.base_h) * 0.95
        self.zoom = max(0.1, z)
        self.pan_x = (cw - self.base_w * self.zoom) / 2
        self.pan_y = (ch - self.base_h * self.zoom) / 2
        self.zoom_scale.set(int(self.zoom * 100))
        self.redraw()

    def force_refresh(self):
        if self.volume is None:
            return
        self.extract_slice()
        self._base_dirty = True
        self.redraw()
        self.update_info_panel()

    # -------------------------------------------------------------- tools
    def on_tool_change(self):
        self.temp = None
        t = self.var_tool.get()
        cur = {"pan": "fleur", "window": "sb_h_double_arrow", "probe": "crosshair",
               "line": "tcross", "poly": "tcross", "pencil": "pencil",
               "hl": "pencil", "erase": "dotbox", "magic": "X_cursor"}.get(t, "crosshair")
        try:
            self.canvas.configure(cursor=cur)
        except Exception:
            self.canvas.configure(cursor="crosshair")
        self.redraw()

    def set_pen_color(self, c):
        self.var_pencolor.set(c)
        self.pen_prev.configure(bg=c)

    def pick_pen_color(self):
        c = colorchooser.askcolor(color=self.var_pencolor.get())[1]
        if c:
            self.set_pen_color(c)

    def clear_slice(self):
        self.annotations[self._key()] = []
        self.redraw()
        self.log("cleared annotations on this slice")

    # -------------------------------------------------------------- canvas events
    def _bind_canvas(self):
        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_b1_press)
        c.bind("<B1-Motion>", self.on_b1_motion)
        c.bind("<ButtonRelease-1>", self.on_b1_release)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<ButtonPress-3>", self.on_right)
        c.bind("<ButtonPress-2>", self.on_mid_press)
        c.bind("<B2-Motion>", self.on_mid_motion)
        c.bind("<Motion>", self.on_motion)
        c.bind("<MouseWheel>", self.on_wheel)          # Windows / macOS
        c.bind("<Button-4>", lambda e: self.on_wheel(e, 120))   # Linux up
        c.bind("<Button-5>", lambda e: self.on_wheel(e, -120))  # Linux down
        c.bind("<Configure>", lambda e: self.redraw())
        self.root.bind("<Escape>", lambda e: self._cancel_temp())
        self.root.bind("<Prior>", lambda e: self._step_slice(-1))   # PageUp
        self.root.bind("<Next>", lambda e: self._step_slice(1))     # PageDown

    def _step_slice(self, d):
        if self.volume is None:
            return
        self.idx = max(0, min(self._axis_len() - 1, self.idx + d))
        self.slice_scale.set(self.idx)

    def _cancel_temp(self):
        self.temp = None
        self.redraw()

    # ---- jump to voxel (X,Y,Z) ----
    def vox_to_disp(self, x, y, z):
        """Map a volume voxel (x,y,z) to (col, row, slice_index) for the current
        axis, accounting for the default and user flips used by extract_slice."""
        Z, Y, X = self.volume.shape
        x = int(np.clip(x, 0, X - 1)); y = int(np.clip(y, 0, Y - 1)); z = int(np.clip(z, 0, Z - 1))
        a = self.var_axis.get()
        if a == self.AXES[0]:                    # Axial: slice=z, col=x, row=y
            idx, col, row, W, H = z, x, y, X, Y
        elif a == self.AXES[1]:                  # Coronal: slice=y (flipud default)
            idx, col, row, W, H = y, x, (Z - 1) - z, X, Z
        else:                                    # Sagittal: slice=x (flipud default)
            idx, col, row, W, H = x, y, (Z - 1) - z, Y, Z
        if self.var_flip_h.get():
            col = (W - 1) - col
        if self.var_flip_v.get():
            row = (H - 1) - row
        return col, row, idx

    def jump_to(self):
        if self.volume is None:
            self.log("No DICOM loaded.")
            return
        try:
            x = int(float(self.var_x.get())); y = int(float(self.var_y.get()))
            z = int(float(self.var_z.get()))
        except ValueError:
            self.log("Invalid coordinate: X, Y, Z must be integers.")
            return
        col, row, idx = self.vox_to_disp(x, y, z)
        self.idx = idx
        self.slice_scale.set(idx)                # -> on_slice (extract + redraw)
        self.extract_slice()
        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())
        self.pan_x = cw / 2 - (col + 0.5) * self.zoom
        self.pan_y = ch / 2 - (row + 0.5) * self.ar * self.zoom
        self.marker = (self.var_axis.get(), idx, col, row)
        self._base_dirty = True
        self.redraw()
        hu = int(self.cur_slice[int(np.clip(row, 0, self.cur_slice.shape[0] - 1)),
                                int(np.clip(col, 0, self.cur_slice.shape[1] - 1))])
        self.log("jump to X%d Y%d Z%d  ->  HU %d  (%s)"
                 % (x, y, z, hu, classify_name(hu, self.materials)))

    def _clamp_img(self, cx, ry):
        cx = min(max(cx, 0), self.cur_slice.shape[1] - 1)
        ry = min(max(ry, 0), self.cur_slice.shape[0] - 1)
        return cx, ry

    def on_motion(self, e):
        if self.volume is None:
            return
        cx, ry = self.canvas_to_img(e.x, e.y)
        if 0 <= cx < self.cur_slice.shape[1] and 0 <= ry < self.cur_slice.shape[0]:
            hu = int(self.cur_slice[int(ry), int(cx)])
            mat = classify_name(hu, self.materials)
            self.var_status.set("col %d  row %d  slice %d   |   HU = %d   |   material: %s"
                                % (int(cx), int(ry), self.idx, hu, mat))
        else:
            self.var_status.set("(outside image)")
        # live polygon rubber-band
        if self.temp and self.temp["t"] == "poly":
            self.temp["cur"] = self.canvas_to_img(e.x, e.y)
            self.draw_annotations()

    def on_wheel(self, e, delta=None):
        if self.volume is None:
            return
        d = delta if delta is not None else e.delta
        shift = bool(e.state & 0x0001)          # Shift held?
        if shift:                                # Shift + wheel -> zoom to cursor
            factor = 1.15 if d > 0 else (1 / 1.15)
            self.set_zoom(self.zoom * factor, e.x, e.y)
        else:                                    # plain wheel -> scroll slices
            self._step_slice(-1 if d > 0 else 1)

    def on_mid_press(self, e):
        self._pan_ref = (e.x, e.y, self.pan_x, self.pan_y)

    def on_mid_motion(self, e):
        if hasattr(self, "_pan_ref"):
            x0, y0, px, py = self._pan_ref
            self.pan_x = px + (e.x - x0)
            self.pan_y = py + (e.y - y0)
            self.redraw()

    # ---- left button dispatch ----
    def on_b1_press(self, e):
        if self.volume is None:
            return
        t = self.var_tool.get()
        cx, ry = self._clamp_img(*self.canvas_to_img(e.x, e.y))
        if t == "pan":
            self._pan_ref = (e.x, e.y, self.pan_x, self.pan_y)
        elif t == "window":
            self._win_ref = (e.x, e.y, self.wl, self.ww)
        elif t == "probe":
            hu = int(self.cur_slice[int(ry), int(cx)])
            self.log("PROBE  col %d row %d slice %d  ->  HU %d  (%s)"
                     % (int(cx), int(ry), self.idx, hu, classify_name(hu, self.materials)))
        elif t == "line":
            self.temp = {"t": "line", "p0": (cx, ry), "p1": (cx, ry)}
        elif t == "poly":
            self._poly_click(cx, ry)
        elif t in ("pencil", "hl"):
            self.temp = {"t": t, "pts": [(cx, ry)]}
        elif t == "erase":
            self._erase_at(cx, ry)
        elif t == "magic":
            self._magic_erase()

    def on_b1_motion(self, e):
        if self.volume is None:
            return
        t = self.var_tool.get()
        cx, ry = self._clamp_img(*self.canvas_to_img(e.x, e.y))
        if t == "pan" and hasattr(self, "_pan_ref"):
            x0, y0, px, py = self._pan_ref
            self.pan_x = px + (e.x - x0); self.pan_y = py + (e.y - y0)
            self.redraw()
        elif t == "window" and hasattr(self, "_win_ref"):
            x0, y0, wl0, ww0 = self._win_ref
            self.wl = wl0 + (e.x - x0) * 2.0
            self.ww = max(1.0, ww0 + (y0 - e.y) * 2.0)
            self.wl_scale.set(self.wl); self.ww_scale.set(self.ww)
            self._base_dirty = True; self.redraw()
        elif t == "line" and self.temp:
            self.temp["p1"] = (cx, ry); self.draw_annotations()
        elif t in ("pencil", "hl") and self.temp:
            self.temp["pts"].append((cx, ry)); self.draw_annotations()
        elif t == "erase":
            self._erase_at(cx, ry)

    def on_b1_release(self, e):
        if self.volume is None:
            return
        t = self.var_tool.get()
        if t == "line" and self.temp:
            p0, p1 = self.temp["p0"], self.temp["p1"]
            if p0 != p1:
                self._list().append({"t": "line", "p": [p0, p1],
                                     "label": self._line_label(p0, p1)})
                self.log("line  %s" % self._line_label(p0, p1))
            self.temp = None
            self.redraw()
        elif t in ("pencil", "hl") and self.temp:
            if len(self.temp["pts"]) >= 2:
                self._list().append({"t": t, "pts": self.temp["pts"],
                                     "color": self.var_pencolor.get(),
                                     "w": self.var_penwidth.get()})
            self.temp = None
            self.redraw()

    # ---- polygon handling ----
    def _poly_click(self, cx, ry):
        if not self.temp or self.temp["t"] != "poly":
            self.temp = {"t": "poly", "v": [(cx, ry)], "cur": (cx, ry)}
            self.draw_annotations()
            return
        vs = self.temp["v"]
        # close if clicking near the first vertex
        x0, y0 = self.img_to_canvas(*vs[0])
        cxs, cys = self.img_to_canvas(cx, ry)
        if len(vs) >= 3 and math.hypot(cxs - x0, cys - y0) < 12:
            self._close_poly()
        else:
            vs.append((cx, ry))
            self.temp["cur"] = (cx, ry)
            self.draw_annotations()

    def _close_poly(self):
        vs = self.temp["v"]
        if len(vs) >= 3:
            self._list().append({"t": "poly", "v": list(vs),
                                 "color": self._next_poly_color(),
                                 "label": self._poly_label(vs)})
            self.log("polygon  %s  (%d vertices)" % (self._poly_label(vs), len(vs)))
        self.temp = None
        self.redraw()

    def on_double(self, e):
        if self.temp and self.temp["t"] == "poly" and len(self.temp["v"]) >= 3:
            self._close_poly()

    def on_right(self, e):
        if self.temp and self.temp["t"] == "poly" and len(self.temp["v"]) >= 3:
            self._close_poly()

    # ---- erasers ----
    def _erase_at(self, cx, ry):
        r = max(4.0, 8.0 / self.zoom)
        lst = self._list()
        keep = []
        removed = 0
        for st in lst:
            hit = False
            if st["t"] in ("pencil", "hl"):
                for px, py in st["pts"]:
                    if math.hypot(px - cx, py - ry) < r:
                        hit = True; break
            elif st["t"] == "line":
                for px, py in st["p"]:
                    if math.hypot(px - cx, py - ry) < r:
                        hit = True; break
            elif st["t"] == "poly":
                for px, py in st["v"]:
                    if math.hypot(px - cx, py - ry) < r:
                        hit = True; break
            if hit:
                removed += 1
            else:
                keep.append(st)
        if removed:
            self.annotations[self._key()] = keep
            self.redraw()

    def _magic_erase(self):
        lst = self._list()
        for i in range(len(lst) - 1, -1, -1):
            if lst[i]["t"] in ("pencil", "hl"):
                del lst[i]
                self.redraw()
                self.log("magic eraser: removed last stroke")
                return
        self.log("magic eraser: no stroke to remove", verbose=True)

    # -------------------------------------------------------------- info panel
    def update_info_panel(self):
        if self.volume is None:
            self.info_lbl.configure(text="No DICOM loaded.")
            return
        m = self.meta
        sy, sx = m["PixelSpacing"]
        npix = self.volume.shape[0] * self.volume.shape[1] * self.volume.shape[2]
        lines = [
            "── DICOM INFO ─────────────",
            "Patient : %s  %s  %s" % (m["PatientID"], m["PatientSex"], m["PatientAge"]),
            "Study   : %s" % m["StudyDescription"],
            "Series  : %s" % m["SeriesDescription"],
            "Modality: %s   %s" % (m["Modality"], m["BodyPart"]),
            "Date    : %s  %s" % (m["StudyDate"], m["AcquisitionTime"]),
            "Maker   : %s" % m["Manufacturer"],
            "Kernel  : %s   Pos: %s" % (m["Kernel"], m["PatientPosition"]),
            "──────────────────────────",
            "Matrix  : %d × %d × %d  (%s vox)" % (
                self.volume.shape[2], self.volume.shape[1], self.volume.shape[0],
                "{:,}".format(npix)),
            "Pixel   : %.3f × %.3f mm   Slice: %.3f mm" % (sx, sy, self.spacing[0]),
            "HU range: %d … %d" % (int(self.volume.min()), int(self.volume.max())),
            "Window  : C %g / W %g" % (self.wl, self.ww),
            "Photom. : %s   Bits: %s" % (m["Photometric"], m["BitsStored"]),
            "── CT technique ──────────",
            "kVp %s   mA %s   mAs %s" % (m["KVP"], m["TubeCurrent_mA"], m["Exposure_mAs"]),
            "Exp %s ms   CTDIvol %s" % (m["ExposureTime_ms"], m["CTDIvol"]),
            "Tilt %s   Contrast %s" % (m["GantryTilt"], m["Contrast"]),
        ]
        self.info_lbl.configure(text="\n".join(str(x) for x in lines))

    # -------------------------------------------------------------- material editor
    def open_material_editor(self):
        win = tk.Toplevel(self.root)
        win.title("HU → material table")
        win.geometry("420x420")
        win.transient(self.root)

        head = ttk.Frame(win, padding=6); head.pack(fill="x")
        for i, t in enumerate(("Material", "HU low", "HU high")):
            ttk.Label(head, text=t, width=14 if i == 0 else 8,
                      font=("", 9, "bold")).grid(row=0, column=i, sticky="w")

        body_outer = ttk.Frame(win); body_outer.pack(fill="both", expand=True)
        bc = tk.Canvas(body_outer, highlightthickness=0)
        sb = ttk.Scrollbar(body_outer, orient="vertical", command=bc.yview)
        body = ttk.Frame(bc)
        body.bind("<Configure>", lambda e: bc.configure(scrollregion=bc.bbox("all")))
        bc.create_window((0, 0), window=body, anchor="nw")
        bc.configure(yscrollcommand=sb.set)
        bc.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        rows = []

        def add_row(mat=None):
            mat = mat or {"name": "New", "low": 0, "high": 100}
            r = ttk.Frame(body); r.pack(fill="x", pady=1)
            ne = ttk.Entry(r, width=16); ne.insert(0, mat["name"]); ne.grid(row=0, column=0, padx=2)
            le = ttk.Entry(r, width=8); le.insert(0, str(mat["low"])); le.grid(row=0, column=1, padx=2)
            he = ttk.Entry(r, width=8); he.insert(0, str(mat["high"])); he.grid(row=0, column=2, padx=2)
            btn = ttk.Button(r, text="✕", width=3,
                             command=lambda: (r.destroy(), rows.remove(entry)))
            btn.grid(row=0, column=3, padx=2)
            entry = (ne, le, he, r)
            rows.append(entry)

        for m in self.materials:
            add_row(m)

        bar = ttk.Frame(win, padding=6); bar.pack(fill="x")
        ttk.Button(bar, text="+ Add row", command=lambda: add_row()).pack(side="left")
        ttk.Button(bar, text="Reset defaults",
                   command=lambda: [r[3].destroy() for r in rows] or rows.clear()
                   or [add_row(m) for m in DEFAULT_MATERIALS]).pack(side="left", padx=4)

        def apply():
            new = []
            try:
                for ne, le, he, _ in rows:
                    new.append({"name": ne.get().strip() or "?",
                                "low": float(le.get()), "high": float(he.get())})
            except ValueError:
                messagebox.showerror("Invalid", "HU low / high must be numbers.", parent=win)
                return
            new.sort(key=lambda m: m["low"])
            self.materials = new
            self._base_dirty = True
            self.redraw()
            self.update_info_panel()
            self.log("material table updated (%d classes)" % len(new))
            win.destroy()

        ttk.Button(bar, text="Apply", command=apply).pack(side="right")
        ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="right", padx=4)


# =============================================================================
#  main
# =============================================================================

def main():
    root = tk.Tk()
    app = DicomViewer(root)
    # allow "python dicom_viewer.py <folder>"
    if len(sys.argv) > 1:
        p = sys.argv[1]
        root.after(200, lambda: app._load(p if os.path.isdir(p) else [p]))
    root.mainloop()


if __name__ == "__main__":
    main()
