import cv2
import numpy as np

def compare_images(img1, img2, method="mse", grayscale=True):
    """
    Compares the similarity between two images using image comparison methods.

    Args:
        img1: First image (NumPy array).
        img2: Second image (NumPy array).
        method: Image comparison method.
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

# TODO check if this works properly
def stitch_images(source, test, threshold=0.75, grayscale=True):
    """
    Stitches two images together based on feature matching and a similarity threshold.

    Args:
        source: The first image (NumPy array).
        test: The second image (NumPy array).
        threshold: The threshold for acceptable feature matches (ratio test). 
                   Lower values mean more strict matching. Defaults to 0.75.

    Returns:
        The stitched image (NumPy array) if successful, otherwise None.
    """
    if not grayscale:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(source, None)
    kp2, des2 = sift.detectAndCompute(test, None)

    # Use a Brute-Force matcher with KNN matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # Check if enough matches are found
    if len(good_matches) > 10:  # Adjust this value based on your needs
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the dimensions of the images
        h1, w1 = source.shape[:2]
        h2, w2 = test.shape[:2]

        # Warp the second image to align with the first
        result = cv2.warpPerspective(test, M, (w1 + w2, h1))

        # Overlay the first image onto the warped second image
        result[0:h1, 0:w1] = source

        return result
    else:
        print("Not enough matching features found for stitching.")
        return None