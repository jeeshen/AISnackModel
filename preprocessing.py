import cv2
import numpy as np
from typing import Tuple


def read_image(path: str) -> np.ndarray:
    """Read an image from disk in BGR format (as used by OpenCV)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return img


def resize_image(
    img: np.ndarray,
    scale: float = None,
    size: Tuple[int, int] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize image either by a scaling factor or to an explicit size.

    - If `size` is provided, it should be (width, height).
    - Otherwise `scale` is used for both width and height (e.g. 0.5 = half size).
    """
    if size is not None:
        return cv2.resize(img, size, interpolation=interpolation)
    if scale is None:
        raise ValueError("Either `scale` or `size` must be provided.")
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation,
    )


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR color image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_binary(gray_img: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Convert grayscale image to binary using a fixed threshold."""
    _, binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return binary


def denoise_average(gray_img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Average filtering (box blur) to reduce noise."""
    return cv2.blur(gray_img, (ksize, ksize))


def denoise_gaussian(gray_img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Gaussian blur for smoother noise reduction."""
    if ksize % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd.")
    return cv2.GaussianBlur(gray_img, (ksize, ksize), sigma)


def denoise_median(gray_img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filtering, useful for salt-and-pepper noise."""
    if ksize % 2 == 0:
        raise ValueError("Median kernel size must be odd.")
    return cv2.medianBlur(gray_img, ksize)


def morphological_ops(
    img: np.ndarray,
    kernel_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform basic morphological operations on an image:
    erosion, dilation, opening, and closing.

    Returns a tuple: (erosion, dilation, opening, closing).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return erosion, dilation, opening, closing


def canny_edges(
    gray_img: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> np.ndarray:
    """Detect edges using the Canny operator."""
    return cv2.Canny(gray_img, low_threshold, high_threshold)


def harris_corners(
    gray_img: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    thresh_ratio: float = 0.01,
) -> np.ndarray:
    """
    Detect corners using Harris Corner Detector.

    Returns a copy of the original grayscale image with corner points marked.
    """
    gray_float = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    result = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    result[dst > thresh_ratio * dst.max()] = [0, 0, 255]
    return result


def normalize_for_model(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    General-purpose image pre-processing helper (resize + [0, 1] float32).

    WARNING: Do NOT use this function to feed images into the snack classifier.
    That model contains a mobilenet_v2.preprocess_input layer which expects raw
    [0, 255] uint8 values and maps them to [-1, 1] internally.  Passing [0, 1]
    floats here would produce silently wrong predictions.
    Use inference.predict_snack() which handles preprocessing correctly.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized

