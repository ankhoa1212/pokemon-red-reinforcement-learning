import cv2
import numpy as np

from stitching.stitch import stitch_images

CANVAS_HEIGHT = 400
CANVAS_WIDTH = 700


def _make_textured_canvas():
    """Builds a large synthetic scene with plenty of distinct-corner
    texture (random shapes of random size/color/position) so OpenCV's
    feature matcher has real structure to align on. Pure random noise
    is deliberately avoided -- it has no coherent structure across
    overlapping crops, which is closer to two *unrelated* images than
    to two views of the same scene."""
    rng = np.random.RandomState(42)
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
    for _ in range(80):
        x = rng.randint(0, CANVAS_WIDTH)
        y = rng.randint(0, CANVAS_HEIGHT)
        size = rng.randint(8, 30)
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        shape = rng.randint(0, 3)
        if shape == 0:
            cv2.circle(canvas, (x, y), size, color, -1)
        elif shape == 1:
            cv2.rectangle(canvas, (x, y), (x + size, y + size), color, -1)
        else:
            cv2.line(canvas, (x, y), (x + size, y - size), color, 3)
    return canvas


def _write_png(path, image):
    assert cv2.imwrite(str(path), image)
    return path


def _make_overlapping_pair(tmp_path):
    """Two crops of the same textured canvas sharing a wide overlapping
    band -- a realistic stand-in for two genuinely overlapping captures."""
    canvas = _make_textured_canvas()
    img_a = canvas[:, 0:400]
    img_b = canvas[:, 250:650]

    path_a = _write_png(tmp_path / "a.png", img_a)
    path_b = _write_png(tmp_path / "b.png", img_b)
    return path_a, path_b


def _make_non_overlapping_pair(tmp_path):
    """Two crops of the same canvas with a wide gap between them --
    no shared content at all for the feature matcher to align on,
    mirroring the little-to-no-overlap case common on real captures."""
    canvas = _make_textured_canvas()
    img_a = canvas[:, 0:150]
    img_b = canvas[:, 550:700]

    path_a = _write_png(tmp_path / "a.png", img_a)
    path_b = _write_png(tmp_path / "b.png", img_b)
    return path_a, path_b


def test_overlapping_images_stitch_successfully(tmp_path):
    path_a, path_b = _make_overlapping_pair(tmp_path)

    status, pano = stitch_images([path_a, path_b])

    assert status == cv2.Stitcher_OK
    assert pano is not None


def test_zero_paths_returns_non_ok_without_raising():
    status, pano = stitch_images([])

    assert status != cv2.Stitcher_OK
    assert pano is None


def test_single_path_returns_non_ok_without_raising(tmp_path):
    canvas = _make_textured_canvas()
    path_a = _write_png(tmp_path / "a.png", canvas)

    status, pano = stitch_images([path_a])

    assert status != cv2.Stitcher_OK
    assert pano is None


def test_insufficient_overlap_returns_non_ok_without_raising(tmp_path):
    path_a, path_b = _make_non_overlapping_pair(tmp_path)

    status, pano = stitch_images([path_a, path_b])

    assert status != cv2.Stitcher_OK


def test_unreadable_path_is_skipped_and_stitch_still_proceeds(tmp_path):
    path_a, path_b = _make_overlapping_pair(tmp_path)
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not a real png")

    status, pano = stitch_images([path_a, corrupt_path, path_b])

    assert status == cv2.Stitcher_OK
    assert pano is not None


def test_stitch_images_has_no_side_effects_on_disk(tmp_path):
    path_a, path_b = _make_overlapping_pair(tmp_path)
    files_before = sorted(p.name for p in tmp_path.iterdir())

    stitch_images([path_a, path_b])

    files_after = sorted(p.name for p in tmp_path.iterdir())
    assert files_after == files_before
