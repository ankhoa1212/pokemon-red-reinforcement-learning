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


def hash_screen_state(screen: np.ndarray) -> Hashable:
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

    Returns:
        A hashable key (bytes) suitable for use as a dict key.
    """
    downsampled = cv2.resize(
        screen.astype(np.uint8), DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA
    )
    quantized = (downsampled.astype(np.uint32) * QUANTIZATION_LEVELS // 256).astype(np.uint8)
    return quantized.tobytes()


def compare_images(img1, img2, method="mse", grayscale=True):
    """
    Compares the similarity between two images using image comparison methods.

    Args:
        img1: First image (NumPy array).
        img2: Second image (NumPy array).
        method: Image comparison method. Ex: "mse", "template", "correlation", "chi-square", "bhattacharyya"
        grayscale: Whether to convert images to grayscale before comparison.

    Returns:
        Similarity score (float). Higher means more similar.
    """
    if grayscale:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # Resize images to the same size for fair comparison
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    if method == "mse":
        mse = np.mean((img1_resized - img2_resized) ** 2)
        similarity = 1 / (1 + mse)  # Invert for similarity
    elif method == "template":
        res = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, _, _ = cv2.minMaxLoc(res)
        # For TM_SQDIFF_NORMED, lower is better
        min_val = (min_val + 1) / 2  # Shift from [-1,1] to [0,1]
        similarity = 1 - min_val  # Invert for similarity
        # similarity = max_val
    elif method in ['correlation', 'chi-square', 'bhattacharyya']:
        methods = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
        } 
        hist1 = cv2.calcHist([img1_resized], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_resized], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        similarity = cv2.compareHist(hist1, hist2, methods[method])
    return similarity