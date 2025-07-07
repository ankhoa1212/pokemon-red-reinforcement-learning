import os
import cv2

def stitch_images(stitch_dir, stitch_filename='map.png', verbose=1):
    """
    Reads all PNG images from the specified directory, stitches them together, and saves the result as 'map.png'.
    """
    files_in_folder = os.listdir(stitch_dir)
    png_files = [stitch_dir + file for file in files_in_folder if file.endswith('.png')]

    images = []
    for file in png_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
        else:
            if verbose > 0:
                print(f"Warning: Could not load image {file}")

    if not images:
        if verbose > 0:
            print("Error: No images were loaded for stitching.")
    else:
        if verbose > 0:
            print(f"Successfully loaded {len(images)} images.")

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        if verbose > 0:
            print("Image stitching successful!")
        map_save = cv2.imwrite(STITCH_DIR + stitch_filename, pano)
        if map_save and verbose > 0:
            print(f"Stitched image saved as '{stitch_filename}'.")
    else:
        if verbose > 0:
            print(f"Image stitching failed with status code: {status}")