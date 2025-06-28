import cv2
import numpy as np

def stitch_images(image1, image2, threshold=0.75):
    """
    Stitches two images together based on feature matching and a similarity threshold.

    Args:
        image1: The first image (NumPy array).
        image2: The second image (NumPy array).
        threshold: The threshold for acceptable feature matches (ratio test). 
                   Lower values mean more strict matching. Defaults to 0.75.

    Returns:
        The stitched image (NumPy array) if successful, otherwise None.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

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
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Warp the second image to align with the first
        result = cv2.warpPerspective(image2, M, (w1 + w2, h1))

        # Overlay the first image onto the warped second image
        result[0:h1, 0:w1] = image1

        return result
    else:
        print("Not enough matching features found for stitching.")
        return None