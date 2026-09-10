from typing import Hashable

import cv2
import numpy as np

# Target resolution (width, height) the screen is downsampled to before
# hashing. Small enough to tolerate animation/menu noise, large enough to
# distinguish genuinely different layouts. Mirrors Go-Explore's cell-
# definition heuristic; retune here if the hash proves too coarse or fine.
DOWNSAMPLE_SIZE = (8, 8)

# Number of discrete intensity buckets each downsampled pixel is quantized
# into. Fewer levels tolerate more animation noise but risk collapsing
# distinct states together; retune alongside DOWNSAMPLE_SIZE.
QUANTIZATION_LEVELS = 8


def hash_screen_state(
    screen: np.ndarray,
    downsample_size: tuple[int, int] = DOWNSAMPLE_SIZE,
    quantization_levels: int = QUANTIZATION_LEVELS,
) -> Hashable:
    """
    Maps a raw screen array to a stable, hashable state-cell key.

    Downsamples the screen to a small fixed resolution and quantizes pixel
    intensity into a small number of levels, mirroring Go-Explore's
    cell-definition heuristic. This tolerates animation and menu noise
    (small pixel-intensity changes that don't cross a quantization boundary
    hash to the same key) while still distinguishing genuinely different
    game states. State identity is derived only from the rendered screen,
    not from any emulator memory.

    Args:
        screen: Single-channel (grayscale) 2D screen array, e.g. the array
            returned by PokemonRedEnv._get_obs() (pyboy.screen.ndarray[:, :, 0]).
            No grayscale conversion is performed here; the input must already
            be single-channel.
        downsample_size: Target (width, height) the screen is downsampled to
            before hashing. Defaults to DOWNSAMPLE_SIZE.
        quantization_levels: Number of discrete intensity buckets each
            downsampled pixel is quantized into. Defaults to
            QUANTIZATION_LEVELS.

    Returns:
        A hashable key (bytes) suitable for use as a dict key.
    """
    downsampled = cv2.resize(
        screen.astype(np.uint8, copy=False), downsample_size, interpolation=cv2.INTER_AREA
    )
    quantized = (downsampled.astype(np.uint32) * quantization_levels // 256).astype(np.uint8)
    return quantized.tobytes()