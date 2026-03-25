"""
MCP Server for Image Processing Tools.

Extracted from task1.py and task2.py — 12 image processing tools
exposed via the Model Context Protocol (stdio transport).

Usage:
    python mcp_server.py
"""

import os
import uuid
import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Output directory for processed images
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(image: np.ndarray, tag: str = "") -> str:
    """Save a processed image and return its absolute path."""
    name = f"{tag}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, image)
    return path


def _load(image_path: str) -> np.ndarray:
    """Load an image from disk; raise on failure."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img


# ===========================================================================
# Core image‑processing functions (copied from task1.py / task2.py)
# ===========================================================================

def _resize_to_max(image: np.ndarray, max_size: int = 80,
                   interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image
    scale = min(max_size / h, max_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def _whitemask(image: np.ndarray, centers: list, radii: list) -> np.ndarray:
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have the same length")
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy), r in zip(centers, radii):
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, thickness=-1)
    out = np.full_like(image, 255)
    if image.ndim == 2:
        out[mask == 255] = image[mask == 255]
    else:
        out[mask == 255, :] = image[mask == 255, :]
    return out


def _gridmask(image: np.ndarray, xp: list = None, yp: list = None,
              angle: list = None, polar: bool = False) -> np.ndarray:
    xp = xp or []
    yp = yp or []
    angle = angle or []
    out = image.copy()
    h, w = out.shape[:2]
    guide_v = (255, 0, 255)
    if not polar:
        for xline in xp:
            cv2.line(out, (xline, 0), (xline, h - 1), guide_v, 1)
        for yline in yp:
            cv2.line(out, (0, yline), (w - 1, yline), guide_v, 1)
    else:
        cx, cy = w // 2, h // 2
        R = int(np.hypot(w, h))
        for deg in angle:
            rad = np.deg2rad(deg)
            x2 = int(round(cx + R * np.cos(rad)))
            y2 = int(round(cy - R * np.sin(rad)))
            x3 = int(round(cx - R * np.cos(rad)))
            y3 = int(round(cy + R * np.sin(rad)))
            cv2.line(out, (cx, cy), (x2, y2), guide_v, 1)
            cv2.line(out, (cx, cy), (x3, y3), guide_v, 1)
    return out


def _crop_image(image: np.ndarray, bbox_list: list) -> list[np.ndarray]:
    if image is None:
        return []
    if bbox_list and isinstance(bbox_list[0], (int, float)):
        bbox_list = [bbox_list]
    h, w = image.shape[:2]
    results = [image]
    for bbox in bbox_list:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x1 >= x2 or y1 >= y2:
            continue
        results.append(image[y1:y2, x1:x2].copy())
    return results


def _near_white_to_binary(image: np.ndarray, white_threshold: int = 110,
                          distance_threshold: float = None) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Requires a BGR color image (H, W, 3)")
    img = image.astype(np.int16)
    if distance_threshold is None:
        mask = np.all(img >= white_threshold, axis=2)
    else:
        dist = np.linalg.norm(img - 255, axis=2)
        mask = dist <= distance_threshold
    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    binary[mask] = 255
    return binary


def _near_red_to_binary(image: np.ndarray, red_threshold: int = 110,
                        distance_threshold: float = None) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Requires a BGR color image (H, W, 3)")
    img = image.astype(np.int16)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if distance_threshold is None:
        redness = r - np.maximum(g, b)
        mask = (r >= red_threshold) & (redness >= 5)
    else:
        red_ref = np.array([0, 0, 255], dtype=np.int16)
        dist = np.linalg.norm(img - red_ref, axis=2)
        mask = dist <= distance_threshold
    bw3 = np.full_like(image, 255, dtype=np.uint8)
    bw3[mask] = (0, 0, 0)
    return bw3


def _laplacian_edge_enhance(image: np.ndarray, ksize: int = 3,
                            alpha: float = 1.2) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    edge_map = cv2.convertScaleAbs(lap)
    edge_map = cv2.convertScaleAbs(edge_map, alpha=alpha)
    if image.ndim == 3:
        edge_3c = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.subtract(image, edge_3c)
    else:
        enhanced = cv2.subtract(image, edge_map)
    return enhanced


def _enhance_luminance_contrast(
    image: np.ndarray,
    clahe_clip: float = 4.2, clahe_grid: tuple = (8, 8),
    detail_gain: float = 2.2, low_pct: float = 0.8, high_pct: float = 99.2,
    sigmoid_k: float = 8.0
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Requires a BGR color image (H, W, 3)")
    img_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    l1 = clahe.apply(l)
    blur = cv2.GaussianBlur(l1, (0, 0), sigmaX=1.2, sigmaY=1.2)
    l2 = cv2.addWeighted(l1, 1.0 + float(detail_gain), blur, -float(detail_gain), 0)
    l2 = np.clip(l2, 0, 255).astype(np.uint8)
    lo = np.percentile(l2, float(low_pct))
    hi = np.percentile(l2, float(high_pct))
    if hi <= lo + 1e-6:
        l3 = l2.astype(np.float32)
    else:
        l3 = (l2.astype(np.float32) - lo) * (255.0 / (hi - lo))
        l3 = np.clip(l3, 0, 255)
    x = l3 / 255.0
    y = 1.0 / (1.0 + np.exp(-float(sigmoid_k) * (x - 0.5)))
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    l_out = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)
    return out


def _draw_three_equal_spacing_vertical_lines(
    image: np.ndarray, x_a: int, x_b: int, thickness: int = 1
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    guide_v = (255, 0, 255)
    x1, x2 = int(round(x_a)), int(round(x_b))
    if x1 == x2:
        raise ValueError("x_a and x_b cannot be identical")
    x3 = x2 + (x2 - x1)
    for x in (x1, x2, x3):
        if 0 <= x < w:
            color = guide_v if out.ndim == 3 else 255
            cv2.line(out, (x, 0), (x, h - 1), color, int(thickness))
    return out


def _reversed_blur_mask(image: np.ndarray, centers: list, radii: list,
                        blur_strength: int = 21, reversed: int = 1) -> np.ndarray:
    if blur_strength % 2 == 0:
        blur_strength += 1
    if not isinstance(centers, list):
        centers = [centers]
    if isinstance(radii, (int, float)):
        radii = [radii]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for center, radius in zip(centers, radii):
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    if reversed == 0:
        mask = 255 - mask
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_normalized = mask_blurred.astype(float) / 255.0
    if len(image.shape) == 3:
        mask_normalized = mask_normalized[:, :, np.newaxis]
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    result = (image * mask_normalized + blurred_image * (1 - mask_normalized)).astype(np.uint8)
    return result


def _enhance_contrast(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
    return enhanced


def _draw_rectangle(image: np.ndarray, bboxes: list,
                    color: tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    if not bboxes:
        return image
    out = image.copy()
    if isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


# ===========================================================================
# MCP Server
# ===========================================================================

mcp = FastMCP(
    "Image Processing Tools",
    instructions=(
        "A collection of 12 image processing tools for visual analysis tasks "
        "including color comparison, size measurement, line geometry judgment, "
        "edge enhancement, contrast enhancement, and more. "
        "All tools accept an image file path and return the processed image file path."
    ),
)


@mcp.tool()
def resize_image(image_path: str, max_size: int = 80) -> str:
    """Resize an image so its longest side fits within max_size pixels.

    Args:
        image_path: Path to the input image file.
        max_size: Maximum allowed dimension (width or height). Default 80.

    Returns:
        Path to the resized image.
    """
    img = _load(image_path)
    result = _resize_to_max(img, max_size=max_size)
    return _save(result, "resize")


@mcp.tool()
def whitemask(image_path: str, centers: list[list[int]], radii: list[int]) -> str:
    """Keep only specified circular regions and make everything else white.

    Useful for isolating small local regions for color comparison.

    Args:
        image_path: Path to the input image file.
        centers: List of circle centers, e.g. [[x1, y1], [x2, y2]].
        radii: List of radii for each circle, e.g. [r1, r2]. Should be < 20 for small targets.

    Returns:
        Path to the masked image.
    """
    img = _load(image_path)
    result = _whitemask(img, centers, radii)
    return _save(result, "whitemask")


@mcp.tool()
def gridmask(image_path: str, xp: list[int] = None, yp: list[int] = None,
             angle: list[float] = None, polar: bool = False) -> str:
    """Draw guide lines on the image for measurement and alignment.

    In Cartesian mode (polar=False): draw vertical lines at xp positions and
    horizontal lines at yp positions.
    In polar mode (polar=True): draw rays from image center at given angles
    (0 deg = right, counterclockwise positive).

    Args:
        image_path: Path to the input image file.
        xp: X-coordinates for vertical lines (Cartesian mode).
        yp: Y-coordinates for horizontal lines (Cartesian mode).
        angle: Angles in degrees for polar rays.
        polar: If True, use polar mode; if False, use Cartesian grid mode.

    Returns:
        Path to the image with guide lines.
    """
    img = _load(image_path)
    result = _gridmask(img, xp=xp, yp=yp, angle=angle, polar=polar)
    return _save(result, "gridmask")


@mcp.tool()
def crop_image(image_path: str, bbox_list: list[list[int]]) -> list[str]:
    """Crop the image to one or more bounding boxes.

    Returns the original image as the first element, followed by each cropped region.

    Args:
        image_path: Path to the input image file.
        bbox_list: List of bounding boxes, e.g. [[x_min, y_min, x_max, y_max], ...].

    Returns:
        List of file paths: [original_image, crop_1, crop_2, ...].
    """
    img = _load(image_path)
    results = _crop_image(img, bbox_list)
    paths = []
    for i, r in enumerate(results):
        tag = "crop_orig" if i == 0 else f"crop_{i}"
        paths.append(_save(r, tag))
    return paths


@mcp.tool()
def near_white_to_binary(image_path: str, white_threshold: int = 110,
                         distance_threshold: float = None) -> str:
    """Convert near-white pixels to white (255) and all others to black (0).

    Useful for isolating near-white regions from a colorful image.

    Args:
        image_path: Path to the input BGR color image.
        white_threshold: Minimum value for each channel to be considered white (default 110).
        distance_threshold: If set, use Euclidean distance to pure white instead of per-channel threshold.

    Returns:
        Path to the binary mask image.
    """
    img = _load(image_path)
    result = _near_white_to_binary(img, white_threshold, distance_threshold)
    return _save(result, "white_binary")


@mcp.tool()
def near_red_to_binary(image_path: str, red_threshold: int = 110,
                       distance_threshold: float = None) -> str:
    """Isolate near-red pixels: red becomes black, everything else becomes white.

    Useful for extracting red lines/regions for geometric judgment.

    Args:
        image_path: Path to the input BGR color image.
        red_threshold: Minimum red channel value (default 110).
        distance_threshold: If set, use Euclidean distance to pure red instead.

    Returns:
        Path to the binary (3-channel BGR) image.
    """
    img = _load(image_path)
    result = _near_red_to_binary(img, red_threshold, distance_threshold)
    return _save(result, "red_binary")


@mcp.tool()
def laplacian_edge_enhance(image_path: str, ksize: int = 3,
                           alpha: float = 1.2) -> str:
    """Enhance edges using a Laplacian filter.

    Makes edges darker and more visible for line/shape structure analysis.

    Args:
        image_path: Path to the input image.
        ksize: Laplacian kernel size (default 3).
        alpha: Edge amplification factor (default 1.2).

    Returns:
        Path to the edge-enhanced image.
    """
    img = _load(image_path)
    result = _laplacian_edge_enhance(img, ksize=ksize, alpha=alpha)
    return _save(result, "edge_enhance")


@mcp.tool()
def enhance_luminance_contrast(image_path: str, clahe_clip: float = 4.2,
                               detail_gain: float = 2.2,
                               sigmoid_k: float = 8.0) -> str:
    """Strongly enhance luminance contrast using CLAHE + detail sharpening + S-curve.

    Enlarges subtle light-dark differences within similar hues. Best for
    color comparison tasks.

    Args:
        image_path: Path to the input BGR color image.
        clahe_clip: CLAHE clip limit (default 4.2).
        detail_gain: High-frequency detail gain (default 2.2).
        sigmoid_k: S-curve steepness for mid-tone separation (default 8.0).

    Returns:
        Path to the contrast-enhanced image.
    """
    img = _load(image_path)
    result = _enhance_luminance_contrast(img, clahe_clip=clahe_clip,
                                         detail_gain=detail_gain,
                                         sigmoid_k=sigmoid_k)
    return _save(result, "lum_contrast")


@mcp.tool()
def draw_equal_spacing_lines(image_path: str, x_a: int, x_b: int,
                             thickness: int = 1) -> str:
    """Draw three equally spaced vertical guide lines.

    Given x-coordinates of lines A and B, automatically computes C = B + (B - A).

    Args:
        image_path: Path to the input image.
        x_a: X-coordinate of the first vertical line (marker A).
        x_b: X-coordinate of the second vertical line (marker B).
        thickness: Line thickness (default 1).

    Returns:
        Path to the image with three guide lines.
    """
    img = _load(image_path)
    result = _draw_three_equal_spacing_vertical_lines(img, x_a, x_b, thickness)
    return _save(result, "equal_lines")


@mcp.tool()
def reversed_blur_mask(image_path: str, centers: list[list[int]],
                       radii: list[int], blur_strength: int = 21,
                       reversed: int = 1) -> str:
    """Apply selective blur based on circular regions.

    reversed=1 (default): circles stay clear, background blurred (highlight targets).
    reversed=0: circles blurred, background clear (for color-blind pattern recognition).

    Args:
        image_path: Path to the input image.
        centers: List of circle centers [[x, y], ...].
        radii: List of radii [r1, r2, ...].
        blur_strength: Gaussian blur kernel size, must be odd (default 21).
        reversed: 1 = clear inside, 0 = blur inside.

    Returns:
        Path to the selectively blurred image.
    """
    img = _load(image_path)
    result = _reversed_blur_mask(img, centers, radii, blur_strength, reversed)
    return _save(result, "blur_mask")


@mcp.tool()
def enhance_contrast(image_path: str, clip_limit: float = 2.0,
                     tile_grid_size_h: int = 8,
                     tile_grid_size_w: int = 8) -> str:
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Helpful for revealing hidden patterns in color blindness tests or low-contrast images.

    Args:
        image_path: Path to the input image (BGR or grayscale).
        clip_limit: Contrast limiting threshold (default 2.0).
        tile_grid_size_h: Grid height for histogram equalization (default 8).
        tile_grid_size_w: Grid width for histogram equalization (default 8).

    Returns:
        Path to the contrast-enhanced image.
    """
    img = _load(image_path)
    result = _enhance_contrast(img, clip_limit, (tile_grid_size_h, tile_grid_size_w))
    return _save(result, "contrast")


@mcp.tool()
def draw_rectangle(image_path: str, bboxes: list[list[int]],
                   color_b: int = 0, color_g: int = 0, color_r: int = 255,
                   thickness: int = 2) -> str:
    """Draw highlight rectangles on the image.

    Useful for marking regions of interest without hiding background context.

    Args:
        image_path: Path to the input image.
        bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...].
        color_b: Blue channel of rectangle color (default 0).
        color_g: Green channel of rectangle color (default 0).
        color_r: Red channel of rectangle color (default 255).
        thickness: Rectangle border thickness (default 2).

    Returns:
        Path to the image with rectangles drawn.
    """
    img = _load(image_path)
    result = _draw_rectangle(img, bboxes, color=(color_b, color_g, color_r),
                             thickness=thickness)
    return _save(result, "rectangle")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
