import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from PIL import Image as PILImage


def get_detection_mask_image(sample):
    detections = getattr(sample.sam3_segmentations, "detections", None)
    detection = detections[0] if detections else None
    mask = getattr(detection, "mask", None)
    if mask is not None:
        return PILImage.fromarray((mask * 255).astype(np.uint8))
    return None


def get_segmented_bbox_image(sample):
    image = PILImage.open(sample.filepath).convert("RGBA")
    img_w, img_h = image.size

    detections = getattr(sample.sam3_segmentations, "detections", None)
    if not detections:
        print("No detections found.")
        return None

    detection = detections[0]
    mask = getattr(detection, "mask", None)
    bbox = getattr(detection, "bounding_box", None)

    if mask is None or bbox is None:
        print("Missing mask or bounding box.")
        return None

    # Convert normalized bbox → pixel coords
    x, y, w, h = bbox
    x1 = int(x * img_w)
    y1 = int(y * img_h)
    x2 = int((x + w) * img_w)
    y2 = int((y + h) * img_h)

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Crop image to bbox
    image_cropped = image.crop((x1, y1, x2, y2))

    # Resize mask to bbox size (NOT full image size)
    mask_resized = np.array(
        PILImage.fromarray((mask * 255).astype(np.uint8))
        .resize((bbox_w, bbox_h), resample=PILImage.NEAREST)
    ).astype(bool)

    image_np = np.array(image_cropped)

    # Apply mask
    image_np[~mask_resized] = [0, 0, 0, 0]  # transparent background

    return PILImage.fromarray(image_np)


import cv2
import numpy as np
from PIL import Image as PILImage

def canny_edge_detection(
    image_path=None,
    image=None,
    low_threshold=100,
    high_threshold=200,
    blur_kernel_size=5
):
    if image_path is None and image is None:
        raise ValueError("Provide either image_path or image")

    # Load image
    if image_path is not None:
        image = cv2.imread(image_path)
    else:
        if isinstance(image, PILImage):
            image = np.array(image)
        if image.shape[-1] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Optional Gaussian blur (reduces noise)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return PILImage.fromarray(edges)