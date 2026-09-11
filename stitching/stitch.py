import cv2


def stitch_images(image_paths):
    """
    Reads the given images and stitches them together into one panorama.

    Takes an explicit list of image paths (Path or str) rather than scanning
    a directory, and returns the raw (status, pano) pair from cv2.Stitcher
    instead of printing or writing output -- callers decide what to do with
    either. Unreadable files are skipped, not fatal, mirroring cv2.imread's
    None-on-failure behavior.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)

    return status, pano
