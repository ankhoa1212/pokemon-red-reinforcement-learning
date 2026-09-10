import numpy as np

from image_checker import hash_screen_state

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
    base_value = 150  # mid-bucket for QUANTIZATION_LEVELS=8 (bucket 150 // 32 == 4)
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
