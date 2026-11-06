"""Unit tests for LiteDicom's GUI-free core: DICOM loading and the HU helpers.

All DICOM data is synthetic and generated on the fly, so no patient data and no
sample files are needed.   Run:  python -m pytest
"""
import random
import warnings

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import LiteDicom as ld


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def write_slice(path, z, stored, *, slope=1.0, intercept=-1024.0, pixel_spacing=(0.8, 0.9),
                thickness=5.0, instance=1, shape=None):
    """Write one synthetic CT slice. `stored` is a scalar or 2-D array of stored pixel values."""
    stored = np.asarray(stored)
    if stored.ndim == 0:
        stored = np.full(shape or (6, 8), int(stored))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientID = "SYNTH"
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = stored.shape
    ds.PixelSpacing = list(pixel_spacing)                    # [row spacing, column spacing]
    ds.SliceThickness = thickness
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]          # axial -> normal is +z
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelData = stored.astype("<u2").tobytes()
    with warnings.catch_warnings():                          # pydicom 2.x needs these, 3.x deprecates them
        warnings.simplefilter("ignore")
        ds.is_little_endian = True
        ds.is_implicit_VR = False
    ds.save_as(str(path))
    return path


def make_series(folder, zs, *, shuffle=False, hu_of=lambda k: 100 * k, **kw):
    """Slice k sits at zs[k] and has a constant HU of hu_of(k) (stored = HU + 1024)."""
    order = list(range(len(zs)))
    if shuffle:
        random.Random(1).shuffle(order)
    for n, k in enumerate(order):
        write_slice(folder / f"img_{n:03d}.dcm", zs[k], hu_of(k) + 1024,
                    instance=random.Random(k).randint(1, 999), **kw)
    return folder


# ----------------------------------------------------------------------------
# load_dicom_series
# ----------------------------------------------------------------------------
def test_volume_shape_and_hounsfield_conversion(tmp_path):
    make_series(tmp_path, [0, 3, 6, 9])
    vol, spacing, meta, n_ok, n_skipped = ld.load_dicom_series(str(tmp_path))
    assert vol.shape == (4, 6, 8) and vol.dtype == np.int16          # [z, y, x]
    assert [int(vol[k].mean()) for k in range(4)] == [0, 100, 200, 300]   # stored*slope + intercept
    assert (n_ok, n_skipped) == (4, 0)
    assert (meta["Rows"], meta["Cols"], meta["Slices"], meta["Modality"]) == (6, 8, 4, "CT")


def test_slices_are_ordered_by_position_not_by_file_name_or_instance_number(tmp_path):
    make_series(tmp_path, [0, 3, 6, 9, 12], shuffle=True)
    vol, *_ = ld.load_dicom_series(str(tmp_path))
    assert [int(vol[k].mean()) for k in range(5)] == [0, 100, 200, 300, 400]


def test_slice_spacing_comes_from_positions_not_from_the_SliceThickness_tag(tmp_path):
    # Overlapping reconstruction: slices 3.4 mm apart but SliceThickness says 5.0 mm.
    make_series(tmp_path, [0, 3.4, 6.8, 10.2], thickness=5.0)
    _, (sz, sy, sx), meta, *_ = ld.load_dicom_series(str(tmp_path))
    assert sz == pytest.approx(3.4)
    assert (sy, sx) == pytest.approx((0.8, 0.9))                      # PixelSpacing = [row, col]
    assert meta["SliceSpacing"] == pytest.approx(3.4)
    assert meta["SliceThickness"] == 5.0


def test_repeated_slice_positions_do_not_drag_the_spacing_down(tmp_path):
    # e.g. a duplicated localizer / repeated acquisition: zero gaps must be ignored,
    # otherwise the median of [0, 0, 3, 3] would be 1.5 instead of 3.
    make_series(tmp_path, [0, 0, 0, 3, 6])
    _, (sz, *_), *_ = ld.load_dicom_series(str(tmp_path))
    assert sz == pytest.approx(3.0)


def test_one_large_gap_does_not_skew_the_spacing(tmp_path):
    make_series(tmp_path, [0, 3, 6, 9, 30])
    _, (sz, *_), *_ = ld.load_dicom_series(str(tmp_path))
    assert sz == pytest.approx(3.0)                                   # median, not mean


def test_rescale_slope_is_applied_and_values_are_clipped_to_int16(tmp_path):
    write_slice(tmp_path / "a.dcm", 0, 1500, slope=2.0, intercept=-1000.0)     # 2000 HU
    write_slice(tmp_path / "b.dcm", 1, 65535, slope=2.0, intercept=-1000.0)    # 130070 -> clipped
    vol, *_ = ld.load_dicom_series(str(tmp_path))
    assert int(vol[0].max()) == 2000
    assert int(vol[1].max()) == 32767


def test_non_dicom_and_corrupt_files_are_skipped_not_fatal(tmp_path):
    make_series(tmp_path, [0, 3, 6])
    (tmp_path / "notes.txt").write_text("not a dicom file")
    (tmp_path / "junk.dcm").write_bytes(bytes(random.Random(0).randrange(256) for _ in range(500)))
    messages = []
    vol, _, _, n_ok, n_skipped = ld.load_dicom_series(str(tmp_path), progress=messages.append)
    assert vol.shape[0] == 3 and n_ok == 3 and n_skipped == 2
    assert any("Done" in m for m in messages)


def test_slice_with_different_matrix_size_is_skipped(tmp_path):
    write_slice(tmp_path / "a.dcm", 0, 1024)
    write_slice(tmp_path / "b.dcm", 3, 1124, shape=(5, 5))           # odd one out, in the middle
    write_slice(tmp_path / "c.dcm", 6, 1224)
    vol, _, _, n_ok, n_skipped = ld.load_dicom_series(str(tmp_path))
    assert vol.shape == (2, 6, 8) and n_ok == 2 and n_skipped == 1


def test_accepts_an_explicit_list_of_files(tmp_path):
    make_series(tmp_path, [0, 3, 6])
    files = sorted(str(p) for p in tmp_path.glob("*.dcm"))[:2]
    vol, *_ = ld.load_dicom_series(files)
    assert vol.shape[0] == 2


def test_empty_folder_returns_none(tmp_path):
    assert ld.load_dicom_series(str(tmp_path)) == (None, None, None, 0, 0)


# ----------------------------------------------------------------------------
# HU -> material helpers
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("hu,name", [
    (-1000, "Air"), (-950, "Lung"), (-500, "Adipose / Fat"), (0, "Water / CSF"),
    (40, "Soft tissue"), (300, "Trabecular bone"), (1000, "Cortical bone"), (3000, "Metal / Dense"),
])
def test_classify_name_uses_half_open_ranges(hu, name):
    assert ld.classify_name(hu, ld.DEFAULT_MATERIALS) == name


def test_classify_indices_matches_classify_name_for_in_range_values():
    values = np.array([-1000, -950, -500, -50, 0, 15, 40, 120, 500, 1499, 1500, 3000])
    idx = ld.classify_indices(values, ld.DEFAULT_MATERIALS)
    names = [ld.DEFAULT_MATERIALS[i]["name"] for i in idx]
    assert names == [ld.classify_name(int(v), ld.DEFAULT_MATERIALS) for v in values]


def test_default_material_table_is_contiguous():
    mats = ld.DEFAULT_MATERIALS
    assert all(a["high"] == b["low"] for a, b in zip(mats, mats[1:]))
    assert all(m["low"] < m["high"] for m in mats)


def test_polygon_area():
    assert ld.polygon_area([(0, 0), (2, 0), (2, 2), (0, 2)]) == pytest.approx(4.0)
    assert ld.polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6.0)
    assert ld.polygon_area([(0, 2), (2, 2), (2, 0), (0, 0)]) == pytest.approx(4.0)   # winding-independent
    assert ld.polygon_area([(0, 0), (1, 1)]) == 0.0


def test_colormap_luts():
    luts = ld.get_luts()
    assert {"Grayscale", "Bone", "Hot", "Jet"} <= set(luts)
    for lut in luts.values():
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
    gray = luts["Grayscale"]
    assert tuple(gray[0]) == (0, 0, 0) and tuple(gray[-1]) == (255, 255, 255)
    assert (np.diff(gray[:, 0].astype(int)) >= 0).all()


def test_material_palette_cycles_through_the_base_palette():
    pal = ld.material_palette("Vivid", 20)
    assert pal.shape == (20, 3) and pal.dtype == np.uint8
    base = len(ld._PALETTES["Vivid"])
    assert tuple(pal[base]) == tuple(pal[0])
    assert ld.material_palette("no-such-palette", 3).shape == (3, 3)      # falls back to Vivid
