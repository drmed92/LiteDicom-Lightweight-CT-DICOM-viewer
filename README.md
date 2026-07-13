# LiteDicom

A lightweight, single-file DICOM viewer and voxel inspector for CT / MR series.
No heavyweight frameworks — just Python + Tkinter, with HU windowing, orthogonal
views, material auto-colouring, measuring and annotation tools.


![Main window](scrshot1.png)
![Main window](scrshot2.png)

---

## Features

### Core
- **Single `.py` file**, runs on Windows 10 (and Linux/macOS) — just `python LiteDicom.py`.
- **Auto-installs prerequisites** (`numpy`, `pillow`, `pydicom`) on first run and tells you it's doing so.
- **Loads a whole DICOM series** from a folder (walked recursively) into a 3-D Hounsfield-Unit volume, or single files.
- Slices **sorted spatially** by `ImagePositionPatient` (falls back to `InstanceNumber`).
- **HU conversion** via `RescaleSlope` / `RescaleIntercept`.
- **Non-blocking threaded loading** with live progress in the log; **non-DICOM, corrupt, or undecodable files are skipped and reported** instead of crashing.

### Viewing
- **Orthogonal views:** Axial (XY), Coronal (XZ), Sagittal (YZ), switchable on the fly.
- **Aspect-correct rendering** using in-plane pixel spacing and slice spacing.
- **Flip H / Flip V** and an **Update view** button for awkward acquisitions.
- **Slice navigation:** full-width bottom slider, **mouse-wheel scroll**, and PageUp / PageDown.
- **Zoom:** slider, **Shift + mouse-wheel** (zooms to cursor), plus **Fit** and **1:1**. Viewport-cropped rendering keeps high zoom fast.
- **Pan:** middle-mouse drag anywhere, or the Pan tool with left-drag.
- **Go to voxel (X, Y, Z):** jumps to the slice, centres the view, and drops a crosshair marker showing the HU and material at that point.

### Windowing & colouring
- **HU windowing** (Level / Width sliders) plus a **Window-drag** tool.
- **Radiology presets:** Soft tissue, Lung, Bone, Brain, Mediastinum, Abdomen, Liver, Full range.
- **Continuous colormaps** (pure NumPy, no matplotlib): Grayscale, Bone, Hot, Jet, Cool, Rainbow, Viridis, Inferno.
- **Auto-colour by material:** HU → material classification with 6 palettes (Vivid, Pastel, Earth, Neon, Warm, Cool); brightness is modulated by the window so anatomy stays visible.
- **Editable HU → material table** (add / remove / edit ranges) with Schneider-style defaults: air, lung, adipose, water/CSF, soft tissue, trabecular bone, cortical bone, metal.

### Voxel probing & DICOM info
- **Live readout** of coordinates, **HU value**, and estimated **material** on hover.
- **Probe** tool logs the HU at a clicked point.
- **Bottom-left info panel** with the fields radiologists scan for:
  - *Identity / study:* patient ID, sex, age, study & series description, modality, body part, date/time, manufacturer, kernel, patient position
  - *Geometry:* matrix + voxel count, pixel spacing, slice spacing, HU range, window
  - *Pixel:* photometric interpretation, bits stored
  - *CT technique:* kVp, mA, mAs, exposure time, CTDIvol, gantry tilt, contrast agent
- **Orientation letters** (R/L/A/P/S/I) on the canvas edges and a corner overlay showing WL / WW / zoom / slice.

### Measurement & annotation
- **Measure line** — distance in mm (px if spacing is unknown).
- **Measure area (polygon)** — click vertices; it auto-closes, trims any dangling segment, fills with a **unique non-repeating 50%-transparent colour**, and reports the enclosed area in mm² / px².
- **Pencil** (colour choices + width) and **Highlighter** (semi-transparent).
- **Eraser** (removes strokes near the cursor) and **Magic eraser** (removes the entire last stroke).
- Annotations are stored **per slice and per axis**, so they reappear as you navigate; plus a **Clear this slice** button, 9 quick colour swatches, and a custom colour picker.

### UX
- **Verbose log** toggle with a live log panel.
- **Scrollable control panel** (mouse wheel or scrollbar) and **screen-resolution-aware** window sizing that tightens the layout on short screens so the tools stay reachable.
- Launch straight into a dataset: `python LiteDicom.py C:\path\to\series`.

---

## Requirements

- Python 3.8+
- `numpy`, `pillow`, `pydicom` (auto-installed on first run)
- Tkinter (bundled with the standard Windows/macOS Python installer; on Linux: `sudo apt install python3-tk`)
- *Optional, for compressed DICOM:* `pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm`

## Install & run

```bash
pip install pydicom numpy pillow
python LiteDicom.py
# or open a series directly:
python LiteDicom.py /path/to/dicom/folder
```

## Windows executable (.exe)

Users who don't have Python can run a standalone **`LiteDicom.exe`** — no install required. There are two ways to produce it (a Windows machine is required to build a Windows exe; it can't be cross-built on Linux/macOS):

**A. Build it yourself (one click).** On Windows, put `build_exe.bat` next to `LiteDicom.py` and double-click it. It installs PyInstaller + dependencies and produces `dist\LiteDicom.exe`. Or manually:

```bash
pip install pyinstaller pydicom numpy pillow
pyinstaller --onefile --windowed --name LiteDicom --collect-all pydicom LiteDicom.py
```

**B. Let GitHub build it.** Copy `build-windows-exe.yml` into `.github/workflows/` in your repo. Then pushing a version tag (e.g. `git tag v1.0.0 && git push --tags`) builds the exe on a Windows runner and **attaches it to that GitHub Release** automatically; you can also trigger it manually from the Actions tab, which uploads the exe as a downloadable artifact. This means you can ship a `.exe` without owning a Windows machine.

> To also support compressed DICOM in the exe, add `pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg` to the pip install and `--collect-all pylibjpeg` to the PyInstaller command.

## Controls

| Action | Control |
|---|---|
| Scroll through slices | Mouse wheel over image · PageUp / PageDown · bottom slider |
| Zoom | **Shift + wheel** (to cursor) · zoom slider · Fit / 1:1 |
| Pan | Middle-mouse drag · Pan tool + left-drag |
| Window level/width | Sliders · presets · Window-drag tool |
| Close a polygon | Click near the first vertex · double-click · right-click |
| Cancel a polygon | Esc |
| Scroll the tools panel | Mouse wheel over the left panel |

## Notes

- Orientation letters follow the usual radiological convention but are **not** derived per-dataset from the direction cosines, so unusual acquisitions may need the Flip H/V toggles.
- Transparency uses Tk's `gray50` stipple (Tk has no true alpha), which reads as ~50% but is not a real composite.
- The HU → material table is intended for **visualisation and quick estimation**, not dosimetry-grade segmentation.

## Credits

A concept by **Ahmed I.** — ahmedalj@pm.me
