import numpy as np

from image_checker import DOWNSAMPLE_SIZE, QUANTIZATION_LEVELS, hash_screen_state

# Typical Game Boy screen resolution (rows, cols) used to build synthetic
# single-channel frames for these tests.
SCREEN_HEIGHT = 144
SCREEN_WIDTH = 160


def make_frame(fill_value: int) -> np.ndarray:
    """Builds a synthetic single-channel screen filled with one value."""
    return np.full((SCREEN_HEIGHT, SCREEN_WIDTH), fill_value, dtype=np.uint8)


def test_same_frame_hashes_to_same_key():
    frame = make_frame(150)

    key1 = hash_screen_state(frame)
    key2 = hash_screen_state(frame)

    assert key1 == key2


def test_small_animation_noise_does_not_change_key():
    base_value = 150  # mid-bucket at the default granularity (150 * QUANTIZATION_LEVELS // 256 == 4)
    frame_a = make_frame(base_value)
    frame_b = frame_a.copy()

    # Perturb a small region (simulating a tile animation) by a few
    # intensity units, staying well within the same quantization bucket.
    frame_b[10:13, 10:13] = base_value + 5

    key_a = hash_screen_state(frame_a)
    key_b = hash_screen_state(frame_b)

    assert key_a == key_b


def test_genuinely_different_layouts_hash_to_different_keys():
    # Left half dark, right half light.
    frame_a = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
    frame_a[:, SCREEN_WIDTH // 2:] = 255

    # Top half dark, bottom half light -- a different tile arrangement.
    frame_b = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
    frame_b[SCREEN_HEIGHT // 2:, :] = 255

    key_a = hash_screen_state(frame_a)
    key_b = hash_screen_state(frame_b)

    assert key_a != key_b


def test_all_black_and_all_white_frames_hash_without_error_and_differ():
    black_frame = make_frame(0)
    white_frame = make_frame(255)

    black_key = hash_screen_state(black_frame)
    white_key = hash_screen_state(white_frame)

    assert black_key != white_key


def test_default_arguments_match_module_constants():
    frame = make_frame(150)

    default_key = hash_screen_state(frame)
    explicit_key = hash_screen_state(frame, DOWNSAMPLE_SIZE, QUANTIZATION_LEVELS)

    assert default_key == explicit_key


def test_overriding_downsample_size_changes_hash_for_a_previously_colliding_pair():
    frame_a = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
    frame_b = frame_a.copy()
    # A single-pixel difference collapses away at the default coarse
    # downsample resolution, but survives at a much finer resolution.
    frame_b[0, 0] = 255

    assert hash_screen_state(frame_a) == hash_screen_state(frame_b)
    assert hash_screen_state(
        frame_a, downsample_size=(SCREEN_WIDTH, SCREEN_HEIGHT)
    ) != hash_screen_state(frame_b, downsample_size=(SCREEN_WIDTH, SCREEN_HEIGHT))


def test_overriding_quantization_levels_collapses_frames_default_keeps_distinct():
    frame_a = make_frame(40)
    frame_b = make_frame(70)

    assert hash_screen_state(frame_a) != hash_screen_state(frame_b)
    assert hash_screen_state(frame_a, quantization_levels=2) == hash_screen_state(
        frame_b, quantization_levels=2
    )


def test_quantization_levels_of_one_hashes_without_error():
    frame = make_frame(150)

    key = hash_screen_state(frame, quantization_levels=1)

    assert isinstance(key, bytes)
