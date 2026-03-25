# %%
import argparse
import base64
import os
import sys
import json
import time
import requests
import concurrent.futures
from typing import Optional
import pandas as pd
from tqdm import tqdm
import random
import re
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
# ---------- Default generation parameters ----------
TEMPERATURE = 0.1 
TOP_P = 1
SEED = 42
MAX_TOKENS = 4096

# Gemini Settings
DEFAULT_API_BASE = "YOUR_API_BASE_URL"
DEFAULT_MODEL = "gemini-3.1-pro-preview" 

# %%

def resize_to_max80(image, max_size=80, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image

    scale = min(max_size / h, max_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

def whitemask(image, centers, radii):
    """
    Keep only the regions specified by circles and make everything else white.
    image: ndarray(H,W) or ndarray(H,W,C)
    centers: [(x1,y1), (x2,y2), ...]
    radii: [r1, r2, ...]
    """
    if len(centers) != len(radii):
        raise ValueError("Lengths of centers and radii must be equal")

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (cx, cy), r in zip(centers, radii):
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, thickness=-1)

    out = np.full_like(image, 255)  # Start with pure white
    if image.ndim == 2:
        out[mask == 255] = image[mask == 255]
    else:
        out[mask == 255, :] = image[mask == 255, :]
    return out


def gridmask(image,xp=[],yp=[],angle=[], polar=False):
    """
    polar=False: Divide both horizontal and vertical into 8 segments, draw black grid lines
    polar=True: With image center as origin, draw a black ray every 30 degrees
    """
    out = image.copy()
    h, w = out.shape[:2]
    black = 0 if out.ndim == 2 else (0,) * out.shape[2]
    guide_v = (255, 0, 255)   # Vertical line: neon purple
     
    if not polar:
        for xline in xp:
            cv2.line(out,(xline,0),(xline,h-1),guide_v,1)
        for yline in yp:
            cv2.line(out,(0,yline),(w-1,yline),guide_v,1)
        # 8-part division -> Draw lines at 1/8 ... 7/8 positions
        
    else:
        cx, cy = w // 2, h // 2
        R = int(np.hypot(w, h))  # Long enough to reach the boundary
        for deg in angle:
            rad = np.deg2rad(deg)
            x2 = int(round(cx + R * np.cos(rad)))
            y2 = int(round(cy - R * np.sin(rad))) 
            x3= int(round(cx-R * np.cos(rad)))
            y3=int(round(cy + R * np.sin(rad)))  # Image coordinate y-axis is downwards
            cv2.line(out, (cx, cy), (x2, y2), guide_v, 1)
            cv2.line(out, (cx, cy), (x3, y3), guide_v, 1)
    return out

def crop_image(image, bbox_list):
    """
    Crop the image to the specified multiple bounding boxes.
    Returns the original image and a list of visually cropped sub-images.
    image: ndarray (H, W, C)
    bbox_list: list of lists, e.g. [[x_min, y_min, x_max, y_max], [...]] or [x_min, y_min, x_max, y_max] (single box compatible)
    """
    if image is None: return None
    
    # Compatible with single box input (convert 1D list to 2D)
    if bbox_list and isinstance(bbox_list[0], (int, float)):
        bbox_list = [bbox_list]
        
    h, w = image.shape[:2]
    results = [image] # Always return the original image as the first element
    
    for bbox in bbox_list:
        if len(bbox) != 4: continue
        x1, y1, x2, y2 = map(int, bbox)
        
        # Boundary protection: prevent coordinates from exceeding image scope
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Invalid coordinate check (e.g. x1 >= x2), skip invalid cropping
        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid crop bbox {bbox}, skipping.")
            continue
            
        cropped = image[y1:y2, x1:x2].copy()
        results.append(cropped)

    return results


def near_white_to_binary(image, white_threshold=110, distance_threshold=None):
   
    if image is None:
        return None

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("near_white_to_binary requires a BGR color image (H, W, 3)")

    img = image.astype(np.int16)

    if distance_threshold is None:
        # Strict channel threshold: all three channels close enough to white
        mask = np.all(img >= white_threshold, axis=2)
    else:
        # Euclidean distance threshold: overall color is close to white
        dist = np.linalg.norm(img - 255, axis=2)
        mask = dist <= distance_threshold

    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    binary[mask] = 255
    return binary

def near_red_to_binary(image, red_threshold=110, distance_threshold=None):
    """
    Returns a 3-channel binary image (BGR), identical format to cv2.imread:
    - Close to red -> [0, 0, 0] (black)
    - Others -> [255, 255, 255] (white)
    """
    if image is None:
        return None

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("near_red_to_binary requires a BGR color image (H, W, 3)")

    img = image.astype(np.int16)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    if distance_threshold is None:
        # Use "red dominance" instead of absolute threshold to preserve anti-aliased edges
        # (like pink or dark red transition pixels) to prevent lines from thinning or thickening
        redness = r - np.maximum(g, b)
        # Relax tolerance for edge transition pixels (> 15)
        mask = (r >= red_threshold) & (redness >= 5)
    else:
        # Euclidean distance to pure red (BGR: 0,0,255)
        red_ref = np.array([0, 0, 255], dtype=np.int16)
        dist = np.linalg.norm(img - red_ref, axis=2)
        mask = dist <= distance_threshold

    # Initialize a completely white image (standard uint8 3-channel)
    bw3 = np.full_like(image, 255, dtype=np.uint8)
    
    bw3[mask] = (0, 0, 0)
    
    return bw3

def laplacian_edge_enhance(image, ksize=3, alpha=1.2):
    if image is None:
        return None

    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # [Removed Gaussian noise reduction step]

    # Laplacian edge extraction results in a strong contrast map with [white edges, black background]
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    edge_map = cv2.convertScaleAbs(lap)

    # Amplify edge intensity using alpha
    edge_map = cv2.convertScaleAbs(edge_map, alpha=alpha)

    # Enhance the original image with black edges:
    # Use cv2.subtract so the original image subtracts the edge map (brighter/whiter areas will be subtracted more, becoming black edges)
    if image.ndim == 3:
        edge_3c = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.subtract(image, edge_3c)
    else:
        enhanced = cv2.subtract(image, edge_map)

    return enhanced

def enhance_luminance_contrast(
    image, clahe_clip=4.2, clahe_grid=(8, 8), detail_gain=2.2, low_pct=0.8, high_pct=99.2, sigmoid_k=8.0
):
    """
    Strongly enhance light and dark contrast:
    1) Perform CLAHE on the L channel of LAB color space
    2) Perform high-frequency detail enhancement on the L channel
    3) Percentile stretching + S-curve to expand mid-tone and edge hierarchy differences
    """
    if image is None:
        return None

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("enhance_luminance_contrast requires a BGR color image (H, W, 3)")

    img_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1) Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    l1 = clahe.apply(l)

    # 2) Luminance detail enhancement (unsharp)
    blur = cv2.GaussianBlur(l1, (0, 0), sigmaX=1.2, sigmaY=1.2)
    l2 = cv2.addWeighted(l1, 1.0 + float(detail_gain), blur, -float(detail_gain), 0)
    l2 = np.clip(l2, 0, 255).astype(np.uint8)

    # 3) Percentile stretching to expand dynamic range
    lo = np.percentile(l2, float(low_pct))
    hi = np.percentile(l2, float(high_pct))
    if hi <= lo + 1e-6:
        l3 = l2.astype(np.float32)
    else:
        l3 = (l2.astype(np.float32) - lo) * (255.0 / (hi - lo))
        l3 = np.clip(l3, 0, 255)

    # 4) S-curve to lift separation between dark and light
    x = l3 / 255.0
    y = 1.0 / (1.0 + np.exp(-float(sigmoid_k) * (x - 0.5)))
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    l_out = np.clip(y * 255.0, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)
    return out

def draw_three_equal_spacing_vertical_lines(image, x_a, x_b, thickness=1):
    """
    Draw three equally spaced vertical lines.
    Input the x-coordinates x1, x2 for the first two lines, and the third line is automatically calculated based on equal spacing: x3 = x2 + (x2 - x1).

    Parameters:
    - image: Input image (grayscale or color)
    - x1: x-coordinate of the first line
    - x2: x-coordinate of the second line
    - thickness: line width, default 1
    """
    if image is None:
        return None

    out = image.copy()
    h, w = out.shape[:2]
    guide_v = (255, 0, 255)

    x1 = int(round(x_a))
    x2 = int(round(x_b))

    if x1 == x2:
        raise ValueError("x1 and x2 cannot be identical, otherwise equal spacing cannot be defined")

    x3 = x2 + (x2 - x1)

    for x in (x1, x2, x3):
        if 0 <= x < w:
            color = guide_v if out.ndim == 3 else 255
            cv2.line(out, (x, 0), (x, h - 1), color, int(thickness))

    return out


# %%
"""
Gemini API solver with parallelism (ThreadPoolExecutor).
Based on poolexecutor.ipynb logic but without tools.
"""

# Allow imports from the current directory
sys.path.append(os.getcwd())

try:
    from helper import Solver
except ImportError:
    sys.path.append("")
    from helper import Solver



# ---------- Helpers ----------
from PIL import Image
import io

def encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def encode_image_from_np(image_np) -> str:
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")

def parse_answer(text):
    if not text:
        return None
    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        content = text[start:end].strip()
        if content in ["0", "1"]: 
            return int(content)
        if content in ["A","B","C","D"]:
            return content
    return -1

def call_openai_api(api_key: str, base_url: str, model: str, prompt: str, image_path: str, image=None) -> str:
    """API call using OpenAI Chat Completion format."""
    
    # Uniformly convert image input to a list processing
    if image is None:
        images_to_process = [image_path]
    elif isinstance(image, list):
        images_to_process = image
    else:
        images_to_process = [image]

    base64_images = []
    for img in images_to_process:
        if isinstance(img, np.ndarray):
            b64 = encode_image_from_np(img)
        elif isinstance(img, str):
            # If it is an already encoded base64 string
            if len(img) > 200: # Rough check if it is base64
                b64 = img.replace("\n", "")
            # Otherwise, treat it as a file path
            elif os.path.exists(img):
                b64 = encode_image(img)
            else:
                b64 = img.replace("\n", "")
        else:
            continue
        base64_images.append(b64)

    # Ensure base_url ends correctly for OpenAI endpoint
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Construct messages payload for Vision
    content_list = [{"type": "text", "text": prompt}]
    for b64 in base64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    max_retries = 5  # Increased retries
    for retry in range(max_retries):
        try:
            # Increased timeout to 120s
            # Proxies are conditionally disabled or redacted
            response = requests.post(url, headers=headers, json=payload, timeout=120, proxies={
        "http": "http://YOUR_PROXY_IP:PORT",
        "https": "http://YOUR_PROXY_IP:PORT",
    })
            
            if response.status_code != 200:
                print(f"API Error {response.status_code} (Retry {retry+1}/{max_retries})")
                print(f"Error Details: {response.text}") # <--- Added this line to view detailed error messages
                if response.status_code in [502, 503, 504, 429]:
                    time.sleep(5 * (retry + 1)) # Increased wait time
                    continue
                return None
            
            try:
                data = response.json()
                # Check structure validity before accessing
                if not data or 'choices' not in data or not data['choices']:
                    print(f"Invalid API response (empty choices): {response.text[:100]}")
                    return None
                
                message = data['choices'][0].get('message', {})
                content = message.get('content')
                
                if content is None:
                     print(f"Content is None. Full message: {message}")
                     
                return content
            except Exception as e:
                print(f"JSON Parsing error: {e}, Content: {response.text[:100]}")
                return None
            
        except Exception as e:
            print(f"Request Exception (Retry {retry+1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(5*(retry+1))
            else:
                return None
    return None
   
toolsize=256

class OpenAIAPISolver(Solver):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_API_BASE):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        print(f"Using OpenAI format model: {self.model}")

    def solve(self, image_path: str, prompt: str ,idx) -> int:
        
        image = cv2.imread(image_path)
        index=idx
        if image is None: return -1
        
        # Original Image Handling - Resize FIRST as numpy, THEN encode
        image_np_small =resize_to_max80(image,max_size=toolsize)
        
        image_b64 = encode_image_from_np(image_np_small)

        # Helper to load, resize as numpy, then encode
        def load_resize_encode(path, max_size=400):
            img = cv2.imread(path)
            if img is None: return ""
            img_small = resize_to_max80(img, max_size=max_size)
            return encode_image_from_np(img_small)

        classifyprompt = f"""You are a task classifier for visual questions.
Class definitions:
1 = color   (asks about same/different color, hue, shade, brightness or asks about if boundary exists between different color regions, and  the picture is usually colorful )
2 = size    (asks about bigger/smaller, same size, radius, width, height, area)
3 = line    (asks about line direction, angle, parallel/perpendicular, distance between lines,alignment, crossing, slope,straight or not)
4 = other   (anything not in 1/2/3)

Input question: {prompt}

Rules:
- Output exactly one tag in this format: <class>1</class>
- No extra words."""

        toolusage_common = """
You are in TOOL-USE STAGE.

Hard constraints:
1) Output tool calls only. Do NOT answer the user question.
2) Use at least one tool whenever it is helpful.
3) Parameters must use real pixel coordinates/sizes from the current image. Do NOT normalize or scale coordinates/radii.
4) You may call multiple tools.

Task:
Given the image and classid, choose the correct tool(s) and parameters.

Required reasoning flow:
1) Class analysis: identify classid and map it to the correct tool strategy.
2) Tool execution: output tool call(s) in XML tags.
"""

        toolusage_1 = toolusage_common + """
Forbidden:use tool3 and tool6 at the same time.
Special mandatory rules:
- If the question asks 'Is there an boundary in between every adjecent regions?' use tool3, don't use tool6!
- If the question asks 'same color?' ,use tool0 and tool6

Tool definitions:
Tool 0: whitemask(image, centers, radii)
- Use when: classid == 1 and only a small local region needs color comparison.
- Do not use when: the image is mainly large color bars and the question asks about boundary.
- ATTENTION:you shoul be careful when choosing the centers, don't circle the wrong position, that will lead to wrong answer!!
- Purpose: keep selected circular regions and white out other areas.
- Rules:
  - centers format: [[x1, y1], [x2, y2], ...]
  - radii format: [r1, r2, ...]
  - radii must be < 20
  - Circle only the key object region, not full objects and not background.

Tool 2: crop_image(image, bbox_list)
- Use when: target is too small.
- Purpose: zoom into key comparison region. Return original image AND multiple cropped regions.
- bbox_list format: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], ...]
- Guidance: crop boundary or compared color regions.

Tool 3: near_white_to_binary(image)
- Use when: you need to isolate near-white regions from a colorful image.
- Purpose: convert image to binary mask, near-white pixels become white (255), all other pixels become black (0).
- Output: single-channel binary image (uint8), values are only 0 or 255.

Tool 6: enhance_luminance_contrast(image)
- use when asking same color or not.
- Must use when: classid == 1 (except boundary questions).
- Purpose: enlarge subtle light-dark differences within similar hues while keeping hue as stable as possible.

Output format:
- First provide reasoning for class analysis and parameter choice.
- Then provide tool call(s) only in XML tags.
- Do NOT include <answer></answer>.

Allowed tags:
<tool0>whitemask(image, centers, radii)</tool0>
<tool2>crop_image(image, [[x1, y1, x2, y2], [x3, y3, x4, y4]])</tool2>
<tool3>near_white_to_binary(image)</tool3>
<tool6>enhance_luminance_contrast(image)</tool6>

Example:
The provided classid is 1 and ask about 'the same color?', so I focus on local color regions. I selected one small circular target region and set radius < 20.
<tool0>whitemask(image, [[100, 150]], [18])</tool0>
<tool6>enhance_luminance_contrast(image)</tool6>
"""

        toolusage_2 = toolusage_common + """
Special mandatory rules:
- You MUST use tool1 anda tool2.use tool2 to zoom in the detail and margin of compared objects.

Tool definitions:
Tool 1: gridmask(image, xp=[...], yp=[...], angle=[...], polar=False)
- WARNING: Cannot draw two lines within adjacent 20 pixels
- Must use when: classid == 2.
- Purpose: draw vertical/horizontal helper lines for size measurement (typically tangent or boundary-aligned).

Tool 2: crop_image(image, bbox_list)
- Use when: target is too small.
- Purpose: zoom into key comparison region. Return original image AND multiple cropped regions.
- bbox_list format: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], ...]
- Guidance: crop compared objects region.

Output format:
- First provide reasoning for class analysis and parameter choice.
- Then provide tool call(s) only in XML tags.
- Do NOT include <answer></answer>.

Allowed tags:
<tool1>gridmask(image, xp=[...], yp=[...], angle=[...], polar=...)</tool1>
<tool2>crop_image(image, [[x1, y1, x2, y2], [x3, y3, x4, y4]])</tool2>
"""

        toolusage_3 = toolusage_common + """

Tool definitions:
Tool 1: gridmask(image, xp=[...], yp=[...], angle=[...], polar=False)
- WARNING: Cannot draw two lines within adjacent 20 pixels
- Use when: classid == 3.
- Purpose: draw helper lines for line geometry judgment.
- Extra rules:
  - line means the main straight part only (ignore decorative/slanted end accessories).
  - If judging diagonal alignment or slanted lines, set polar=True and provide an angle list.
  - ATTENTION: Angle convention: 0° points to right direction , consistent with standard polar coordinates. Positive angles rotate counterclockwise(important)!Answer only positive angles!
  - Use polar mode for radial/circular geometric structures.

Tool 2: crop_image(image, bbox_list)
- Must use when: classid == 3 (zoom into line details near tool1 auxiliary lines).
- Purpose: zoom into key comparison region. Return original image AND multiple cropped regions.
- bbox_list format: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], ...]
- Guidance: crop line/column regions for vertical/straight/parallel judgment.

Tool 4: near_red_to_binary(image)
- Use when: the question focuses on red lines/regions (straightness, parallelism, alignment), or red targets are hard to isolate.
- Purpose: isolate near-red pixels into a black/white result for easier geometric judgment.

Tool 5: laplacian_edge_enhance(image)
- Use when: edges or boundaries are blurry and line/shape structure is hard to judge directly.
- Purpose: enhance edge contrast to support line geometry and boundary analysis.

Tool 7: draw_three_equal_spacing_vertical_lines(image, x_a, x_b)
- Use when: the question is exactly 'Are the distances between the vertical markers labeled A–B and B–C equal?'.
- Must use when: the above question appears.
- Do NOT use tool1 (gridmask) for this question.
- Inputs: x_a and x_b are the x coordinates of markers A and B; marker C is inferred by equal spacing.
- Purpose: create three equally spaced vertical guide lines to compare AB and BC distances.

Special mandatory rules:
- If the question is exactly 'Are the distances between the vertical markers labeled A–B and B–C equal?', you MUST use tool7 with x_a and x_b, and MUST NOT use tool1 (gridmask).
- For other classid == 3 questions, you MUST use both tool1 and tool2.
- If the question asks 'is diagonal lines aligned?' (or equivalent diagonal alignment question), set polar=True in tool1 and provide angle values.
- if the question asks 'Are the those red lines straight?', you should prioritize tool4 (red extraction),tool1(grid max) and use tool2 to magnify details around red lines when needed.
- if the question asks 'Are those vertical columns parallel?'  use tool1 and tool5 and tool2(ONLY crop the tops and bottoms of vertical columns).

Output format:
- First provide reasoning for class analysis and parameter choice.
- Then provide tool call(s) only in XML tags.
- Do NOT include <answer></answer>.

Allowed tags:
<tool1>gridmask(image, xp=[...], yp=[...], angle=[...], polar=...)</tool1>
<tool2>crop_image(image, [[x1, y1, x2, y2], [x3, y3, x4, y4]])</tool2>
<tool4>near_red_to_binary(image)</tool4>
<tool5>laplacian_edge_enhance(image)</tool5>
<tool7>draw_three_equal_spacing_vertical_lines(image, x_a=..., x_b=...)</tool7>
"""

        auxline="the neopurple lines are auxiliary lines, they are not the original parts of picture.they are not main body of picture.You should only trust your own auxline and other mark(red tangles),don't trust the original auxline in the pciture,which is usually black and can be deceptive" \
        "the answer of equal size or straight line should be strictly based on the auxline"
        
        # Class 1: Color tasks - focused on brightness, saturation, or hue contrast
        class1prompt = (
            "TASK: COLOR PERCEPTION (ABSOLUTE VALUE ANALYSIS). "
            "WARNING: This image may resemble a famous optical illusion (e.g., Munker-White or Adelson's Checker-shadow,Mach bands).However, the answer is not that simple and the result may not be the normal answer to the illusion"
            "Hints: For 'Is there a boundary between every adjacent region?' questions,ignore 'every' in question. answer yes (1) if there exist multiple balck regions (any white gap between black regions, even one). Answer no (0) only if there only exists one black region."
            "DO NOT give a 'textbook' answer based on experience or common knowledge of illusions. Your intuition is likely to be deceived by the background. You should check the picture itself carefully, instead of relying on the base knowledge of illusion!"
            "REQUIREMENT: You must ignore the surrounding context and focus ONLY on the raw color properties of the target areas. "
            "Compare the targets as if you are measuring their RGB/brightness values pixel by pixel. "
            "The whitemask tool has removed most background; ignore any residual colorful margins. "
            f"Question: {prompt}. Answer with only 0 (no) or 1 (yes)."
        )

        # Class 2: Size tasks - focused on spatial proportions, length, or area ratios
        class2prompt = (
            "TASK: SIZE & SCALE COMPARISON (MEASUREMENT MODE). "
            "WARNING: This image is designed to trigger size-constancy illusions (e.g., Müller-Lyer, Ponzo, or Ebbinghaus illusion). However, the answer is not that simple and the result may not be the normal answer to the illusion"
            "DO NOT guess based on how the objects 'look' relative to the perspective or background. Empirical answers are UNRELIABLE. "
            "STRICT REQUIREMENT: Treat the auxiliary lines as a physical ruler or a coordinate system. "
            "Convert the problem into a geometry task: Do the endpoints of the objects align perfectly with the SAME auxiliary lines? Does the  SAME auxline be tangent to both objects?"
            "Is the pixel-distance between lines identical to the object's length/width?Same size require 2 objects' width and height are the same "
            "Measure meticulously and ignore the distracting background figures. "
            f"Question: {prompt}. Answer with only 0 (no) or 1 (yes)."
        )

        # Class 3: Line tasks - focused on parallelism, length, alignment, or geometry
        class3prompt = (
            "TASK: LINE & GEOMETRY ANALYSIS (STRUCTURAL VERIFICATION). "
            "WARNING: This image may look like a classic geometric illusion (e.g., Hering, Wundt, or Zöllner illusion) where straight lines appear curved or tilted.However, the answer is not that simple and the result may not be the normal answer to the illusion "
            "STRICT REQUIREMENT: Use the provided auxiliary lines as the absolute ground truth for orientation and alignment.  To judge  a line is straight doesn't necessarily mean it need to coincide with auxline, the distance remain constant(parallel) is ok too"
            "the 'lines' mentioned in the question only refers to the straight part of the figure, not including the accessories or slanted parts at both ends"
            "Hints: For 'Are those vertical columns parallel?', ignore the diagonal zebra texture and judge only the outer column edges."
            "Compare each vertical auxline with the column edge at both top and bottom: if the relative gap stays the same, answer yes; if the gap changes, shifts, or intersects, answer no."
            "Hints: When asked 'two black lines of equal length?',check if the purple auxlines to locate the ends are coincident for 2 black lines or not" 
            "Hints: When asked 'Are those red lines straight/parallel?', note that the processed target lines are actually black lines composed of small segments. "
            "Use the purple auxlines as the absolute straight reference. If these black segments are aligned with each other and remain strictly parallel to the auxline, the answer is yes (1). "
            "If the segments are malposed, shifted, or form a ladder/staircase shape that deviates from the auxline, the answer is no (0). "
            "Hints: For the question 'Are the distances between the vertical markers labeled A-B and B-C equal?', treat the three purple vertical guide lines as reference and benchmark,distance between 3 lines is equal,use them to judge distance between  A B C"
             
            f"Question: {prompt}. Answer with only 0 (no) or 1 (yes)."
        )
        
        # Class 4: Optical illusion tasks - the most challenging, rely wholly on logical instructions
        class4prompt = (
            "TASK: OPTICAL ILLUSION CHALLENGE. This task is designed to be visually deceptive. "
            "CRITICAL: Your visual intuition may be wrong. You MUST rely entirely on the auxlines and measurement.  "
            f". Question: {prompt}. Answer with only 0 (no) or 1 (yes)."
        )
        cotprompt="give me the reasoning process and chain of thought before answering the question"
        
        # 1. Classify
        try:
            resp1 = call_openai_api(self.api_key, self.base_url, self.model, classifyprompt.format(prompt=prompt), image_path, image_b64)
            if resp1:
                classify = int(resp1.split("<class>")[1].split("</class>")[0].strip())
            else:
                classify = 4
        except:
            classify = 4

        # 2. Tool usage (Only for classes 1, 2, 3)
        outputimg_np = image_np_small
        processed=image if classify in [2,3] else image_np_small
        eval_env = {
                            "whitemask": whitemask, 
                            "gridmask": gridmask, 
                            "image_np_small":processed, 
                            "image":processed,
                            "crop_image": crop_image, 
                            "near_white_to_binary": near_white_to_binary,
                            "near_red_to_binary":near_red_to_binary,
                            "laplacian_edge_enhance":laplacian_edge_enhance,
                            "enhance_luminance_contrast": enhance_luminance_contrast,
                            "draw_three_equal_spacing_vertical_lines": draw_three_equal_spacing_vertical_lines,
                            "np": np, 
                            "cv2": cv2
                        }
        if classify in [1, 2, 3]:
            try:
                h,w=processed.shape[:2] 
                print(f"{index} tool image {h}{w}\n")
                
                # 1. Call API to get response
                tool_prompts = {1: toolusage_1, 2: toolusage_2, 3: toolusage_3}
                selected_toolusage = tool_prompts.get(classify, toolusage_1)

                tool_resp=call_openai_api(
                    self.api_key, self.base_url, self.model, 
                    f"The classid of the question is {classify},the height and width of image are {h}p,{w}p.Call tool for: question:{prompt}\n tool use guidance:{selected_toolusage}", 
                     image_path,image=processed
                )
                
                
                if tool_resp:
                    try:
                        with open("./cot_trace.txt", "a", encoding="utf-8") as f:
                             f.write(f"=== Index {index} toolresp ===\n{tool_resp}\n\n")
                    except Exception as e:
                         print(f"Error writing CoT: {e}")
                    
                    # 2. Use RegEx to match <tool0>...</tool0> or <tool1>...</tool1>
                    # re.DOTALL allows . to match newlines, [01] matches 0 or 1
                    
                    if("<tool3>"in tool_resp):
                        match = tool_resp.split("<tool3>")[1].split("</tool3>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool4>"in tool_resp):
                        match = tool_resp.split("<tool4>")[1].split("</tool4>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool5>"in tool_resp):
                        match = tool_resp.split("<tool5>")[1].split("</tool5>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool6>"in tool_resp):
                        match = tool_resp.split("<tool6>")[1].split("</tool6>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool7>"in tool_resp):
                        match = tool_resp.split("<tool7>")[1].split("</tool7>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool0>"in tool_resp):
                        match = tool_resp.split("<tool0>")[1].split("</tool0>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool1>"in tool_resp):
                        match = tool_resp.split("<tool1>")[1].split("</tool1>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    if("<tool2>"in tool_resp):
                        match = tool_resp.split("<tool2>")[1].split("</tool2>")[0].strip()
                        processed_list = eval(match,  eval_env) # eval returns a list of [original_image, cut1, cut2...]
                        if isinstance(processed_list, list) and len(processed_list) > 0:
                            processed = processed_list
                            eval_env["image_np_small"] = processed[0] # Take original image to update the environment
                            eval_env["image"] = processed[0]
                        else:
                            processed = processed_list
                            
                    if isinstance(processed, np.ndarray):
                        outputimg_np = processed
                        save_dir = "./processed_images"
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{index}_processed.png")
                        cv2.imwrite(save_path, processed)
                    elif isinstance(processed, list):
                        outputimg_np = processed # Pass list for Final inference
                        save_dir = "./processed_images"
                        os.makedirs(save_dir, exist_ok=True)
                        for i, p_img in enumerate(processed):
                            save_path = os.path.join(save_dir, f"{index}_processed_{i}.png")
                            cv2.imwrite(save_path, p_img)
                    else:
                        print(f"No valid tool tag found in response. Response was: {tool_resp[:100]}...")
                              
            except Exception as e:
                print(f"Tool error: {e}")
                if 'tool_resp' in locals():
                    print(f"Full response: {tool_resp}")
            
        # 3. Final inference
        try:
            if isinstance(outputimg_np, list):
                final_b64 = []
                for img in outputimg_np:
                    if classify in [1]:
                        img = resize_to_max80(img, max_size=128)
                    _, buffer = cv2.imencode(".jpg", img)
                    final_b64.append(base64.b64encode(buffer).decode("utf-8"))
            else:
                if classify in [1]:
                    final_img_np = resize_to_max80(outputimg_np,max_size=128)
                else:
                    final_img_np = outputimg_np
                _, buffer = cv2.imencode(".jpg", final_img_np)
                final_b64 = [base64.b64encode(buffer).decode("utf-8")]

            prompt_map = {1: class1prompt, 2: class2prompt, 3: class3prompt, 4: class4prompt}
            
            # Simplify complex list structure to string for OpenAI
            final_prompt_template = prompt_map.get(classify, class4prompt)
            answerformat="no matter the final answer is number or ABCD, you should give answer between  <answer></answer> dont't include other things like brackets just number or character"
            
            # If there are multiple images, display a prompt
            multi_img_hint = "\nYou are provided with multiple images. The first is the full image, and the following are zoomed-in local crops to help you see detailed structures better.\n" if len(final_b64) > 1 else ""
            final_prompt = auxline + cotprompt+"\n" + multi_img_hint + final_prompt_template.format(prompt=prompt)+answerformat
            
            resp2 = call_openai_api(self.api_key, self.base_url, self.model, final_prompt, "dummy_path_for_log", image=final_b64)
            try:
                with open("./cot_trace.txt", "a", encoding="utf-8") as f:
                    f.write(f"=== Index {index} ===\n{resp2}\n\n")
            except Exception as e:
                print(f"Error writing CoT: {e}")
            return parse_answer(resp2)
        except Exception as e:
            print(f"Final inference error: {e}")
            return -1

    def model_info(self) -> dict:
        return {"model": self.model, "parameters": {"temperature": TEMPERATURE}}

def process_single_row(args):
    solver, idx, row = args
    # Add an outermost try-except block to prevent a single thread from deadlocking the entire pool's future waits
    try:
        # Changed to image_path and prompt to adapt to csv
        result = solver.solve(row["image_path"], row["prompt"] ,idx)
        return idx, result
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, -1

def run_parallel_multikey(api_keys: list, base_url: str, model: str, output_txt: str, output_json: str = "model.json", input_csv=None, max_workers: int = 5, df=None):
    """
    Version supporting multiple API Keys in parallel.
    """
    # 1. Initialize Solvers Pool
    open("./cot_trace.txt", "w", encoding="utf-8")
    solvers = []
    print(f"Initializing {len(api_keys)} solvers...")
    for key in api_keys:
        s = OpenAIAPISolver(api_key=key, base_url=base_url, model=model)
        solvers.append(s)

    # 2. Prepare Data
    if df is None:
        if input_csv:
            df = pd.read_csv(input_csv)
            csv_dir = os.path.dirname(os.path.abspath(input_csv))
            # Compatible with absolute and relative path checks
            if "image_path" in df.columns:
                df["image_path"] = df["image_path"].apply(
                    lambda x: x if os.path.isabs(x) else os.path.join(csv_dir, x)
                )
        else:
            print("Error: No dataframe or input_csv provided.")
            return
    else:
        # Assuming df needs path processing when passed
        data_dir = ""
        if "image" in df.columns and "image_path" not in df.columns:
             df["image_path"] = df["image"].apply(lambda x: x if os.path.isabs(x) else os.path.join(data_dir, x))
    
    # 3. Task Distribution (Round-Robin Solver distribution)
    results = {}
    tasks = []
    
    # Process rows
    for i, (idx, row) in enumerate(df.iterrows()):
          if 200<i <400: 
            assigned_solver = solvers[i % len(solvers)] # Round-Robin distribution
            tasks.append((assigned_solver, idx, row))

    print(f"Total tasks: {len(tasks)} using {len(solvers)} API keys.")

    # 4. Parallel Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_idx = {executor.submit(process_single_row, t): t[1] for t in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks)):
            idx = future_to_idx[future]
            try:
                # Set single task timeout to prevent infinite deadlock
                remote_idx, result = future.result(timeout=240) 
                results[remote_idx] = result
            except concurrent.futures.TimeoutError:
                print(f"Task {idx} timed out.")
                results[idx] = -1
            except Exception as e:
                print(f"Task {idx} failed: {e}")
                results[idx] = -1

    # 5. Save results
    sorted_indices = sorted(results.keys())
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    with open(output_txt, "a") as f:
        for idx in sorted_indices: 
            f.write(f"{idx} {results[idx]}\n")
            
    # Save model info (just take the first solver's)
    if solvers:
        with open(output_json, "w") as f: 
            json.dump(solvers[0].model_info(), f, indent=2)

if __name__ == "__main__":
    # Configure API Keys list
    my_api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2", "YOUR_API_KEY_3", "YOUR_API_KEY_4", "YOUR_API_KEY_5"]
   
    input_csv = "../test.csv"
    output_txt = "../testresult.txt"
    
    base_url1 = "YOUR_API_BASE_URL"
    
    model_name = "gemini-3.1-pro-preview-cli"

    # Call multi-Key parallel function
    # max_workers is recommended to be set to len(my_api_keys) * N, where N is the concurrency supported per Key
    run_parallel_multikey(
        api_keys=my_api_keys,
        base_url=base_url1,
        model=model_name,
        output_txt=output_txt,
        input_csv=input_csv,
        max_workers=10
    )