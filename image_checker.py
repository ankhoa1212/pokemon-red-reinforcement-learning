import cv2
import numpy as np

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