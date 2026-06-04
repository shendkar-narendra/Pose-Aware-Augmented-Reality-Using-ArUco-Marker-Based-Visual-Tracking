import cv2
import numpy as np
import os

# =====================================
# PATHS
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
POSTER_PATH = os.path.join(BASE_DIR, "poster", "poster.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD POSTER (NO ROTATION EVER)
# =====================================
poster = cv2.imread(POSTER_PATH)
if poster is None:
    raise RuntimeError("Poster not found")

ph, pw = poster.shape[:2]
poster_corners = np.array([
    [0, 0],
    [pw, 0],
    [pw, ph],
    [0, ph]
], dtype=np.float32)

# =====================================
# ARUCO DICTIONARIES (CRITICAL)
# =====================================
ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_7X7_50
]

# Canonical marker square
marker_model = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=np.float32)

# =====================================
# PROCESS IMAGES
# =====================================
for filename in sorted(os.listdir(DATA_DIR)):

    if not filename.lower().endswith(".jpg"):
        continue

    image = cv2.imread(os.path.join(DATA_DIR, filename))
    if image is None:
        continue

    corners = None

    # ---------------------------------
    # TRY ALL DICTIONARIES
    # ---------------------------------
    for d in ARUCO_DICTS:
        dictionary = cv2.aruco.getPredefinedDictionary(d)
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            cv2.aruco.DetectorParameters()
        )

        c, ids, _ = detector.detectMarkers(image)

        if ids is not None:
            corners = c
            break

    if corners is None:
        print(f"No marker in {filename}")
        continue

    # ---------------------------------
    # USE CORNERS AS-IS (NO REORDERING)
    # ---------------------------------
    marker_corners = corners[0][0].astype(np.float32)

    # Marker → image homography
    H_marker, _ = cv2.findHomography(marker_model, marker_corners)

    # ---------------------------------
    # SCALE IN MARKER PLANE
    # ---------------------------------
    scale = 4.0

    scaled_marker = np.array([
        [-scale/2, -scale/2],
        [1+scale/2, -scale/2],
        [1+scale/2, 1+scale/2],
        [-scale/2, 1+scale/2]
    ], dtype=np.float32)

    dst_corners = cv2.perspectiveTransform(
        scaled_marker.reshape(-1, 1, 2),
        H_marker
    ).reshape(-1, 2)

    # ---------------------------------
    # POSTER → IMAGE HOMOGRAPHY
    # ---------------------------------
    H_poster, _ = cv2.findHomography(
        poster_corners,
        dst_corners
    )

    warped = cv2.warpPerspective(
        poster,
        H_poster,
        (image.shape[1], image.shape[0])
    )

    # ---------------------------------
    # MASK & OVERLAY
    # ---------------------------------
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_corners.astype(int), 255)
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(image, image, mask=mask_inv)
    foreground = cv2.bitwise_and(warped, warped, mask=mask)

    result = cv2.add(background, foreground)

    # ---------------------------------
    # SAVE
    # ---------------------------------
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), result)
    print(f"Processed {filename}")

print("ALL DONE")
