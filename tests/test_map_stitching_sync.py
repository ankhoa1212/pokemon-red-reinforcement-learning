from pathlib import Path

import cv2
import numpy as np
import pytest

from pokemon_red_env import worker_data_directory
from tensorboard_callback import (
    collect_new_screenshots,
    resolve_current_map,
    write_master_map,
)

CANVAS_HEIGHT = 20
CANVAS_WIDTH = 20


def _make_image(fill_value):
    """A tiny, genuinely-decodable BGR image -- just enough for cv2.imread
    to succeed, standing in for a real screenshot/master map."""
    return np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), fill_value, dtype=np.uint8)


def _write_png(path, fill_value):
    assert cv2.imwrite(str(path), _make_image(fill_value))
    return str(path)


def _write_corrupt(path):
    """A file that exists but cv2.imread cannot decode -- the "present
    but unreadable" case every readability check here must distinguish
    from a genuinely missing file."""
    path.write_bytes(b"not a real png")
    return str(path)


def _save_worker_screenshot(env_data_directory, worker_id, image_directory, filename, fill_value):
    """Mirrors PokemonRedEnv.calculate_fitness's save location: each
    worker gets its own numbered subdirectory under env_data_directory,
    with an image_directory subdirectory beneath that."""
    worker_dir = env_data_directory / str(worker_id) / image_directory
    worker_dir.mkdir(parents=True, exist_ok=True)
    return _write_png(worker_dir / filename, fill_value)


# --- collect_new_screenshots: happy path -------------------------------

def test_collect_new_screenshots_returns_unseen_screenshots_from_every_worker(tmp_path):
    env_data_directory = tmp_path / "env_data"
    path_a = _save_worker_screenshot(env_data_directory, 0, "images", "a.png", 10)
    path_b = _save_worker_screenshot(env_data_directory, 1, "images", "b.png", 20)

    result = collect_new_screenshots(env_data_directory, "images", already_seen=set())

    assert sorted(result) == sorted([path_a, path_b])


def test_collect_new_screenshots_excludes_already_seen_paths(tmp_path):
    env_data_directory = tmp_path / "env_data"
    path_a = _save_worker_screenshot(env_data_directory, 0, "images", "a.png", 10)
    path_b = _save_worker_screenshot(env_data_directory, 1, "images", "b.png", 20)

    result = collect_new_screenshots(
        env_data_directory, "images", already_seen={path_a}
    )

    assert result == [path_b]


def test_collect_new_screenshots_does_not_mutate_already_seen(tmp_path):
    # R4: only the caller advances already_seen, and only on a successful
    # stitch -- collect_new_screenshots itself must be side-effect-free so
    # a failed stitch's batch can be recomputed unchanged on retry.
    env_data_directory = tmp_path / "env_data"
    _save_worker_screenshot(env_data_directory, 0, "images", "a.png", 10)
    already_seen = set()

    collect_new_screenshots(env_data_directory, "images", already_seen)

    assert already_seen == set()


# --- collect_new_screenshots: edge case ---------------------------------

def test_collect_new_screenshots_returns_empty_when_nothing_new(tmp_path):
    env_data_directory = tmp_path / "env_data"
    path_a = _save_worker_screenshot(env_data_directory, 0, "images", "a.png", 10)

    result = collect_new_screenshots(
        env_data_directory, "images", already_seen={path_a}
    )

    assert result == []


def test_collect_new_screenshots_returns_empty_when_no_workers_exist(tmp_path):
    env_data_directory = tmp_path / "env_data"

    result = collect_new_screenshots(env_data_directory, "images", already_seen=set())

    assert result == []


# --- collect_new_screenshots / worker_data_directory: stay in sync ------

def test_collect_new_screenshots_finds_screenshot_saved_via_worker_data_directory(tmp_path):
    """Guards the coupling between the two path-construction sites:
    collect_new_screenshots builds its glob pattern from
    worker_data_directory (see its docstring), so a screenshot saved at
    the exact directory that function returns for a real worker_id must
    be found -- if PokemonRedEnv.__init__'s directory layout ever changes,
    it can only change via worker_data_directory, and this test would
    catch the glob silently going stale right along with it."""
    env_data_directory = str(tmp_path / "env_data") + "/"
    worker_id = "7"
    image_directory = "images"
    worker_dir = Path(worker_data_directory(env_data_directory, worker_id))
    image_dir = worker_dir / image_directory
    image_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = _write_png(image_dir / "a.png", 10)

    result = collect_new_screenshots(env_data_directory, image_directory, already_seen=set())

    assert result == [screenshot_path]


# --- resolve_current_map: happy path / preference order -----------------

def test_resolve_current_map_prefers_primary_when_both_readable(tmp_path):
    map_path = _write_png(tmp_path / "master_map.png", 1)
    backup_path = _write_png(tmp_path / "master_map.prev.png", 2)

    assert resolve_current_map(map_path, backup_path) == map_path


# --- resolve_current_map: edge cases ------------------------------------

def test_resolve_current_map_falls_back_to_backup_when_primary_missing(tmp_path):
    map_path = str(tmp_path / "master_map.png")  # never created
    backup_path = _write_png(tmp_path / "master_map.prev.png", 2)

    assert resolve_current_map(map_path, backup_path) == backup_path


def test_resolve_current_map_falls_back_to_backup_when_primary_corrupt(tmp_path):
    map_path = _write_corrupt(tmp_path / "master_map.png")
    backup_path = _write_png(tmp_path / "master_map.prev.png", 2)

    assert resolve_current_map(map_path, backup_path) == backup_path


def test_resolve_current_map_returns_none_when_both_missing(tmp_path):
    map_path = str(tmp_path / "master_map.png")
    backup_path = str(tmp_path / "master_map.prev.png")

    assert resolve_current_map(map_path, backup_path) is None


def test_resolve_current_map_returns_none_when_both_corrupt(tmp_path):
    map_path = _write_corrupt(tmp_path / "master_map.png")
    backup_path = _write_corrupt(tmp_path / "master_map.prev.png")

    assert resolve_current_map(map_path, backup_path) is None


# --- write_master_map: integration ---------------------------------------

def test_write_master_map_backs_up_old_content_before_atomic_replace(tmp_path):
    map_path = tmp_path / "master_map.png"
    backup_path = tmp_path / "master_map.prev.png"
    old_content = _make_image(1)
    cv2.imwrite(str(map_path), old_content)
    new_content = _make_image(2)

    write_master_map(new_content, map_path, backup_path)

    # The old content was preserved as the backup before being replaced.
    assert np.array_equal(cv2.imread(str(backup_path)), old_content)
    # The new content is now at map_path.
    assert np.array_equal(cv2.imread(str(map_path)), new_content)


# --- write_master_map: edge case ------------------------------------------

def test_write_master_map_skips_backup_when_existing_map_is_corrupt(tmp_path):
    map_path = tmp_path / "master_map.png"
    backup_path = tmp_path / "master_map.prev.png"
    map_path.write_bytes(b"not a real png")
    last_good_backup = _make_image(3)
    cv2.imwrite(str(backup_path), last_good_backup)
    new_content = _make_image(4)

    write_master_map(new_content, map_path, backup_path)

    # The corrupt current map must never overwrite the last known-good
    # backup -- resolve_current_map's fallback depends on it surviving.
    assert np.array_equal(cv2.imread(str(backup_path)), last_good_backup)
    # The new content still lands at map_path.
    assert np.array_equal(cv2.imread(str(map_path)), new_content)


# --- write_master_map: error path / atomicity -----------------------------

def test_write_master_map_leaves_original_untouched_if_replace_is_interrupted(
    tmp_path, monkeypatch
):
    map_path = tmp_path / "master_map.png"
    backup_path = tmp_path / "master_map.prev.png"
    old_content = _make_image(1)
    cv2.imwrite(str(map_path), old_content)
    new_content = _make_image(2)

    def _boom(*args, **kwargs):
        raise OSError("simulated crash before the atomic rename completes")

    monkeypatch.setattr("tensorboard_callback.os.replace", _boom)

    with pytest.raises(OSError):
        write_master_map(new_content, map_path, backup_path)

    # map_path itself must still hold the original content -- the failed
    # replace should only ever have touched the temp file, never the
    # real target.
    assert np.array_equal(cv2.imread(str(map_path)), old_content)

    # The temp file the write staged its new content into is left behind
    # by the simulated crash, and it holds the content that *would* have
    # been swapped in -- direct evidence the write path stages to a temp
    # file rather than writing map_path in place.
    tmp_candidates = list(tmp_path.glob(".master_map.tmp.png"))
    assert len(tmp_candidates) == 1
    assert np.array_equal(cv2.imread(str(tmp_candidates[0])), new_content)


def test_write_master_map_leaves_original_untouched_if_imwrite_fails(
    tmp_path, monkeypatch
):
    # cv2.imwrite returns False on failure (e.g. disk full, permission
    # error) rather than raising -- unlike the os.replace failure above,
    # this must be treated as an ordinary failed-sync case (R13): no
    # exception, and map_path must not be swapped for a file that was
    # never actually written.
    map_path = tmp_path / "master_map.png"
    backup_path = tmp_path / "master_map.prev.png"
    old_content = _make_image(1)
    cv2.imwrite(str(map_path), old_content)
    new_content = _make_image(2)

    monkeypatch.setattr(
        "tensorboard_callback.cv2.imwrite", lambda *args, **kwargs: False
    )

    result = write_master_map(new_content, map_path, backup_path)

    assert result is False
    # map_path itself must still hold the original content -- a failed
    # imwrite must never reach the os.replace swap.
    assert np.array_equal(cv2.imread(str(map_path)), old_content)
