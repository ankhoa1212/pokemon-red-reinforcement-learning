import cv2


def stitch_images(image_paths):
    """
    Reads the given images and stitches them together into one panorama.

    Takes an explicit list of image paths (Path or str) rather than scanning
    a directory, and returns the raw (status, pano) pair from cv2.Stitcher
    instead of printing or writing output -- callers decide what to do with
    either. Unreadable files are skipped, not fatal, mirroring cv2.imread's
    None-on-failure behavior.

    Returns:
        A (status, pano, used_paths) triple. used_paths is the subsequence
        of image_paths that cv2.imread actually decoded and fed into the
        stitch attempt -- i.e. image_paths with the None-reads (unreadable
        or corrupt files) skipped, in the same order. This is reported
        even when status is not OK, and lets a caller distinguish "this
        input was incorporated into the stitch attempt" from "this input
        was skipped this round" independently of the overall stitch
        status -- a path skipped here (e.g. a screenshot caught mid-write)
        may be perfectly readable moments later and should be retried, not
        treated as if it had been processed.
    """
    images = []
    used_paths = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
            used_paths.append(path)

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)

    return status, pano, used_paths
