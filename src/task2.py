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
import numpy as np
import cv2
from PIL import Image
import io

# ---------- Default generation parameters ----------
TEMPERATURE = 0.1 
TOP_P = 1
SEED = 42
MAX_TOKENS = 4096

# # Gemini Settings - Removed user-specific API keys and URLs per instructions
# DEFAULT_API_BASE = "YOUR_API_BASE_URL"
# DEFAULT_MODEL = "gemini-3-flash-preview-cli-mockstream" 
# api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"]


# %%

def reversed_blur_mask(image, centers, radii, blur_strength=21, reversed=1):
    """
    Reversed blur mask: specified circular areas are clear, while the exterior is blurred.
    Supports multiple circles by computing their union.
    
    Args:
        image: Input image (numpy array)
        centers: List of center positions, each element is an (x, y) tuple. Automatic conversion if a single tuple is passed.
        radii: List of circle radii. Automatic conversion if a single integer is passed.
        blur_strength: Blur intensity (must be an odd number)
        reversed: Flag indicating blur direction.
    
    Returns:
        Processed image
    """
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Process parameters: convert to list if not already a list (for backward compatibility)
    if not isinstance(centers, (list, tuple)) or (isinstance(centers, tuple) and len(centers) == 2 and isinstance(centers[0], int)):
        centers = [centers]
    if not isinstance(radii, (list, tuple)) or isinstance(radii, list) and len(radii) > 0 and isinstance(radii[0], (int, float)):
        if isinstance(radii, int):
            radii = [radii]
    
    # Create mask - draw the union of all circles
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw all circles
    for center, radius in zip(centers, radii):
        cv2.circle(mask, tuple(center), int(radius), 255, -1)
    if reversed==0:
        mask=-mask
        mask=mask+255
    # Apply Gaussian blur to mask to achieve smooth transitions
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_normalized = mask_blurred.astype(float) / 255.0
    
    # If it is a color image, expand mask dimensions
    if len(image.shape) == 3:
        mask_normalized = mask_normalized[:, :, np.newaxis]
    
    # Create blurred image
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # Mix original and blurred images: original inside the circle, blurred outside
    result = (image * mask_normalized + blurred_image * (1 - mask_normalized)).astype(np.uint8)
    
    return result

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input image (numpy array, BGR or grayscale)
        clip_limit: Contrast limit threshold, default 2.0
        tile_grid_size: Grid size for histogram equalization, default (8, 8)
    
    Returns:
        Image with enhanced contrast
    """
    # If it is a color image, convert to LAB space and process the L channel
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # Apply directly to grayscale images
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)
        
    return enhanced_image

def whitemask(image, centers, radii):
    """
    Make everything white except for the specified circular regions.
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

    out = np.full_like(image, 255)  # Start pure white
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
        R = int(np.hypot(w, h))  # Long enough to reach boundaries
        for deg in angle:
            rad = np.deg2rad(deg)
            x2 = int(round(cx + R * np.cos(rad)))
            y2 = int(round(cy - R * np.sin(rad))) 
            x3= int(round(cx-R * np.cos(rad)))
            y3=int(round(cy + R * np.sin(rad)))  # Image coordinate y-axis is downwards
            cv2.line(out, (cx, cy), (x2, y2), guide_v, 1)
            cv2.line(out, (cx, cy), (x3, y3), guide_v, 1)
    return out

def crop_image(image, bbox):
    """
    Crop the image to the specified bounding box.
    image: ndarray (H, W, C)
    bbox: [x_min, y_min, x_max, y_max]
    """
    if image is None: return None
    
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    
    # Boundary protection: prevent coordinates from exceeding image scope
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Invalid coordinate check (e.g. x1 >= x2), return original image to prevent errors
    if x1 >= x2 or y1 >= y2:
        print(f"Warning: Invalid crop bbox {bbox}, returning original.")
        return image

    return image[y1:y2, x1:x2].copy()

def draw_rectangle(image, bboxes, color=(0, 0, 255), thickness=2):
    """
    Draw multiple rectangles on the image (usually used for highlighting).
    image: ndarray
    bboxes: [[x_min, y_min, x_max, y_max], ...] (List of lists)
            Also compatible with single bbox: [x_min, y_min, x_max, y_max]
    color: (B, G, R) default Red
    """
    if image is None: return None
    if not bboxes: return image
    
    out = image.copy()
    
    # Check if it is a single bbox (1D list), convert to 2D list if so
    if isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]
        
    for bbox in bboxes:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # cv2.rectangle requires top-left and bottom-right coordinates
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        except Exception as e:
            print(f"Error drawing rectangle {bbox}: {e}")
            continue
            
    return out

def resize_to_max80(image, max_size=80, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image

    scale = min(max_size / h, max_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

# %%
"""
Task execution script.
Based on poolexecutor.ipynb logic but without tools.
"""

# Allow imports from the current directory
sys.path.append(os.getcwd())

try:
    from helper import Solver
except ImportError:
    # ---------------------------
    # PATH UPDATE REQUIRED:
    # Put the root directory path of your project here.
    # ---------------------------
    sys.path.append("YOUR_PROJECT_ROOT_DIR")
    # from helper import Solver

# ---------- Helpers ----------

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
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            return mapping[content]
        
    return -1


def call_openai_api(api_key: str, base_url: str, model: str, prompt: str, image_path: str, image=None) -> str:
    """API call using OpenAI Chat Completion format."""
    
    # --- Fix 1: Ensure base64_image is in string format ---
    if image is None:
        base64_image = encode_image(image_path)
    elif isinstance(image, np.ndarray):
        # If input is a numpy array, encode it to base64 first
        base64_image = encode_image_from_np(image)
    else:
        # Assume it is already a base64 string
        base64_image = image

    # Simple sanitization to prevent newline characters in base64 leading to JSON parsing errors
    if isinstance(base64_image, str):
        base64_image = base64_image.replace("\n", "")

    # Ensure base_url ends correctly for OpenAI endpoint
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Construct messages payload for Vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    max_retries = 5  # Increased retries
    for retry in range(max_retries):
        try:
            # Increased timeout to 150s
            response = requests.post(url, headers=headers, json=payload, verify=False, timeout=150)
            requests.packages.urllib3.disable_warnings() 
            if response.status_code != 200:
                print(f"API Error {response.status_code} (Retry {retry+1}/{max_retries})")
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

# ---------- Solver ----------

# Dummy Solver class for dependency handling if helper.py is unavailable
class Solver:
    pass

class OpenAIAPISolver(Solver):
    # def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_API_BASE):
    def __init__(self, api_key: str, model: str, base_url: str, output_log_path: str = "YOUR_OUTPUT_LOG_PATH/cot_trace.txt", processed_image_dir: str = "YOUR_PROCESSED_IMAGES_DIR"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        # Keep track of paths
        self.output_log_path = output_log_path
        self.processed_image_dir = processed_image_dir
        print(f"Using OpenAI format model: {self.model}")

    def solve(self, image_path, prompt: str ,idx) -> int:
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        # Method 1: Convert to PIL Image
        index=idx
    
        if image is None: return -1
        
        # Original Image Handling - Resize FIRST as numpy, THEN encode
        image_np_small =resize_to_max80(image,max_size=200)
        
        image_b64 = encode_image_from_np(image_np_small)

        # Helper to load, resize as numpy, then encode
        def load_resize_encode(path, max_size=400):
            img = cv2.imread(path)
            if img is None: return ""
            img_small = resize_to_max80(img, max_size=max_size)
            return encode_image_from_np(img_small)

        classifyprompt = f"""You are a task classifier for visual questions.
Class definitions:
1 = counting   (asks about the number(question including how many)of some objects(like fingers))
2 = find_difference    (try ro find the different object or character/figure among a group of similar figures)
3 = color-blind (the image consists of plentiful small color blocks like circle or triangle )
4 = motion    (the image seems like moving and there is motion blur and the question asks about moving or pulsing)
5 = geometry (the image consists of complex lines and geometric figures. question usually looks like:is there a hole in the image? Is this a three-dimensional figure?Can the figure shown in the image exist in the real three-dimensional world? )
6 = real_picture (the picture is usually streetscape or scenery and seems like photograph)
7= size (the picture consists of simple line and circles ,question usually asking about size or length comparition)
8= others(ambiguous task, hard to assigned to the above)
Input question: {prompt}

Rules:
- Output exactly one tag in this format: <class>1</class>
- No extra words

"""
  
        toolusage = """
## Task Objective
Analyze the input image and classid to determine the necessary image processing tool. YOU should not skip this step, use the tool to help analysis.Attention: Just give the tool call, don't give answer in response!!
ATTENTION:you can only use one type of tool once, don't call the same tool more than once!
**Crucial Requirement:** You must provide a detailed step-by-step reasoning (Chain of Thought) BEFORE outputting the tool call. Do not provide the final answer to the user's question yet; only focus on tool selection and parameter calculation.
    You can call multiple tools to better process the picture if you need to.
## Processing Steps (Chain of Thought)
Your output must follow this logical flow:
1. **Class Analysis**: Identify the provided classid and state which tool it maps to.
2. **Visual Reason**: Explain why this specific tool is needed for the given task/image.
3. **Parameter Derivation**: (If using Tool 0) Explain how you calculated the `centers` and `radii` based on the coordinates of the target objects. (If using Tool 1) Explain why polar coordinates are or are not used.
4. **Tool Execution**: Provide the tool call in the specified XML tags.

## Tool Definitions
Attention:Your parameters in the toolcall should base on the real size of the picture you receive,don't normalize or amplify the coordinates or radii !!
- **Tool 0: whitemask(image_np_small, centers, radii)**
    -don't use it when classid ==1
  - Function: Keeps specific circular areas and whites out the rest of the background.Just circle the main body which you need count number of.
  - Requirements: Define `centers` as [[x1, y1], ...] and `radii` as [r1, ...].`radii` should be smaller than 20.
- **Tool 1: gridmask(image_np_small,xp=[x1,x2,x3,...],yp=[y1,y2,y3,....],angle=[degree1,degree2,...] polar=False)**
  - Use when: classid == 2 or 7.
  - Function: Adds helper grid lines when classid==2 or 7.
 (classid==7) you need to give the positions of vertical and horizontal lines  you want to add in xp and yp, mostly tangent to the objects to beeter measure or compare the size
 (classid ==2)  you can also use the tool to seperate a group of similar figures to better locate the position of every objects.
    
-  **Tool 2: crop_image(image, bbox)**
WARNING!!:don't use this tool when classid=1 or 2 ,which requires you to locate the different object in the whole picture
  -must Use when: classid == 6or7  or any task where the target object is too small.
  - Function: Crops the image to the specified bounding box to zoom in on details.
  - Requirement: `bbox` = [x_min, y_min, x_max, y_max].

- **Tool 3: draw_rectangle(image, bboxs, color=(0,0,255), thickness=3)**
  -must Use when: classid ==1 or 2.
  - Function: Draws multiple rectangles (default red) to highlight a specific region without hiding the background context.
  when classid==1 ,draw rectangles on the objects which you need to count nubers of
  when classid==2 ,draw rectangles on the object different from the others
  - Requirement:bbox is a list of lists.`bboxs`=[bbox1,bbox2,...] ,`bbox` = [x_min, y_min, x_max, y_max].

 - **Tool 4: reversed_blur_mask(image, centers, radii, blur_strength=21, reversed=1)**
  -must Use when: classid == 3 (colorblind) .
  - Function: Applies a blur effect to the image based on circular regions.You need to blur the central part to better see the figure in colorblind picture 
    attention: When classid ==3 use reversed=0!
    - `reversed=1` (Default): The area INSIDE the circles remains CLEAR, while the background becomes blurred. Use this to highlight targets.
    - `reversed=0`: when classid ==3 The area INSIDE the circles becomes BLURRED,while the background remains clear.ATTENTION:the raius should be large enough to cover the whole part,you should make radii larger than the part your judge!!  Use this to better see the color blind picture .
  - Parameters:
    - `centers`: List of [x, y] coordinates.
    - `radii`: List of radii.
    - `blur_strength`: Odd integer (e.g., 21) controlling blur intensity.

- **Tool 5: enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8))**
  - Use when: classid == 3 (Color/camouflaged) or when objects are faint/low contrast.
  - Function: Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization). Helpful for revealing hidden patterns in color blindness tests or low-light images.
  - Parameters:
    - `clip_limit`: Threshold for contrast limiting (default 2.0).
    - `tile_grid_size`: Size of grid for histogram equalization (default (8,8)).
## Output Format
[Your detailed reasoning and analysis here]

<tool0>whitemask(image_np_small, centers, radii)</tool0> 
OR 
<tool1>gridmask(image_np_small,xline=[],yline=[],angle=[], polar=...)</tool1>
OR
<tool2>crop_image(image, [x1, y1, x2, y2])</tool2>
OR
<tool3>draw_rectangle(image, [[x1, y1, x2, y2],[x1,y1,x2,y2],...])</tool3>
or ...(tool4,tool5)
---
Example Output:
"The provided classid is 1, which requires focusing on specific objects. To highlight the cells in the center, I have identified the coordinates... therefore:
<tool0>whitemask(image_np_small, [[100, 150]], [50])</tool0>"
"""
        auxline="the neopurple lines are auxiliary lines, they are not the original parts of picture.they are not main body of picture"
        # Class 1: Counting task
        class1prompt = (
            "TASK: PRECISE COUNTING. Your goal is to enumerate specific objects perfectly. "
            "WARNING: Objects may be clustered, overlapping, or small.And objects may not be complete, some parts may be hidden,you need to consider the real size of objects(eg  a dog can't be too long, when the rear part is far from the front part, you should consider as 2 dogs "
            "Attention:this task require you to count the number of real object ,not including  figures similar to your target(eg: asking the number of birds, then you should just count the real bird,the same for other things)"
            ". Do not estimate. "
            "STRICT REQUIREMENT: If you have used the 'whitemask' or 'reversed_blur_mask' tools, focus ONLY on the highlighted/clear areas. "
            "Scan the image systematically (e.g., left-to-right, top-to-bottom). "
            "Ignore background clutter or objects that do not match the target description. "
            f"Question: {prompt}. Answer with only the number or the specific option requested."
        )

        # Class 2: Find Difference task
        class2prompt = (
            "TASK: ANOMALY DETECTION (FIND DIFFERENCE). Focus on detailed comparison. "
            "WARNING: The differences might be very subtle (e.g., slight rotation, missing stroke, color shift). "
            "STRICT REQUIREMENT: Systematically compare every candidate object against the others. "
            "Look for breaks in symmetry or pattern deviations. "
            "If you used 'draw_rectangle', focus deeply on that region to verify the difference. "
            "the question usually ask you to give the coordinate of the different object which usually be among the neatly arranged similar items, you should count the position carefully and use the purple auxlines to help locating"
            f"Question: {prompt}. Answer with the option or location as requested."
        )

        # Class 3: Color-blind / Camouflage detection task
        class3prompt = (
            "TASK: PATTERN RECOGNITION (COLOR/CAMOUFLAGE). Identify hidden shapes/characters in dot patterns. "
            "WARNING: This is likely a color blindness test plate or a camouflage image. "
            "STRICT REQUIREMENT: Rely on the 'enhance_contrast' or 'reversed_blur_mask' tools if applied. "
            "Look for chains of dots with similar hue/saturation that form a structure against the background noise. "
            "Trace the continuity of colors to form the character or shape. The actual result can be characters or numbers or combination of a bunch of characters and numbers "
            f"Question: {prompt}. Answer with the recognizable number, letter, or shape."
        )

        # Class 4: Motion Illusion task
        class4prompt = (
            "TASK: STATIC IMAGE ANALYSIS (MOTION PERCEPTION). "
            "WARNING: This image may contain 'motion illusions' where static patterns appear to move. "
            "STRICT REQUIREMENT: The image is objectively STATIC (a standard 2D image file). "
            "If the question asks 'Is it moving?', the scientific answer is NO, Attention:if the question ask 'Is the image moving or pulsing?' the answer should always be NO too"
            f"Question: {prompt}. Answer specifically based on the visual evidence."
        )

        # Class 5: Geometry Structure task
        class5prompt = (
            "TASK: GEOMETRIC & SPATIAL REASONING. Analyze 3D structures, topology, or holes. "
            "WARNING: Be careful of 'impossible figures' (like Penrose triangles) or perspective tricks.The answer to this question is usually NO(not 3d ,not possible in reality ,no hole) "
            "STRICT REQUIREMENT: Trace the lines carefully. Check if connections are physically possible in 3D space. "
            "If checking for holes , count entrances and exits visible *and* implied behind. "
            f"Question: {prompt}. Answer with logical geometric deduction."
        )

        # Class 6: Real Picture Commonsense task
        class6prompt = (
            "TASK: REAL-WORLD SCENE UNDERSTANDING. Analyze the photo using common sense and physics. "
            "WARNING: Consider perspective, lighting, shadows, and real-world scale. "
            "STRICT REQUIREMENT: Identify objects and their context (indoors/outdoors, season, activity). "
            "your judge should based on category of objects instead their size in the picture(eg: human is  smaller than architecture, a hand is not possible bigger than a human )"
            "you should follow the common sence, this  picture is real not fictional.(eg:people can't touch the roof,a hand can't lift a person,a house can't be lifted by a hand, a person can't kiss the mouth of a statue"
            "there won't be a gap or glacier or lava (something not normal in real life in the middle of a road))"
            "the persons in the picture usually act exaggeratively which is unreliable for judgement "
            "the picture looks strange or bizarre just because the shooting angle and perspective"
            f"Question: {prompt}. Answer based on visual facts and real-world logic."
        )

        # Class 7: Size Comparison task
        class7prompt = (
            "TASK: SIZE & SCALE COMPARISON (MEASUREMENT). "
            "WARNING: This image may look like a classic geometric illusion (e.g., Hering, Wundt, or Zöllner illusion) where straight lines appear curved or tilted.However, the answer is not that simple and the result may not be the normal answer to the illusion "
            "STRICT REQUIREMENT: Use the provided auxiliary lines as the absolute ground truth for orientation and alignment. "
            "Check if the target lines maintain a constant distance from the straight auxiliary lines or if they intersect the grid at identical intervals. "
            "Focus strictly on the geometric structure and ignore all background patterns or interrupting diagonal lines. "
            f"Question: {prompt}. Answer based on strict measurement, not impression."
        )
        
        # Class 8: Miscellaneous task
        class8prompt = (
            "TASK: GENERAL VISUAL ANALYSIS WITH ASSISTANCE. "
            "WARNING: The task category is ambiguous, so you must rely strictly on the visual cues added to the image. "
            "STRICT REQUIREMENT: "
            "1. If purple auxiliary lines (gridmask) are present, use them as the absolute reference for alignment, straightness, or size comparison. "
            "2. If a specific area is highlighted (via whitemask, reversed_blur_mask, or rectangle), IGNORE everything outside that area. Focus 100% on the highlighted content. "
            "3. If the image contains text or numbers, read them carefully. "
            f"Question: {prompt}. Answer based on the highlighted visual evidence."
        )
        
        # Mapping for easy access
        prompt_map = {
            1: class1prompt, 2: class2prompt, 3: class3prompt, 
            4: class4prompt, 5: class5prompt, 6: class6prompt, 
            7: class7prompt, 8:class8prompt
        }

        cotprompt="give me the reasoning process and chain of thought before answering the question"
        # 1. Classify
        try:
            resp1 = call_openai_api(self.api_key, self.base_url, self.model, classifyprompt.format(prompt=prompt), image_path, image_b64)
            if resp1:
                classify = int(resp1.split("<class>")[1].split("</class>")[0].strip())
            else:
                classify = 8
        except:
            classify = 8

        # 2. Tool usage (Only for classes 1, 2, 3)
        outputimg_np = image_np_small
        processed=image if classify in [2,6] else image_np_small
        eval_env = {
                            "whitemask": whitemask, 
                            "gridmask": gridmask, 
                            "image_np_small":processed, 
                            "image":processed, 
                            "crop_image": crop_image, 
                            "reversed_blur_mask":reversed_blur_mask,
                            "enhance_contrast":enhance_contrast,
                            "draw_rectangle": draw_rectangle,
                            "np": np, 
                            "cv2": cv2
                        }
        if classify in range(9) and classify !=0:
            try:
                h,w=image.shape[:2] if classify in[2,6] else image_np_small.shape[:2]
                print(f"{index} tool image {h}{w}\n")
                # Call API to get response
            
                tool_resp=call_openai_api(
                    self.api_key, self.base_url, self.model, 
                    f"the height and width of image are {h}p,{w}p.Call tool for: {prompt}\n{toolusage}", 
                     image_path,image=processed
                )
                
                
                if tool_resp:
                    try:
                        # Ensures the directory exists before logging
                        os.makedirs(os.path.dirname(self.output_log_path), exist_ok=True)
                        with open(self.output_log_path, "a", encoding="utf-8") as f:
                             f.write(f"=== Index {index} toolresp ===\n{tool_resp}\n\n")
                    except Exception as e:
                         print(f"Error writing CoT: {e}")
                    
                    # 2. Use RegEx to match <tool0>...</tool0> or <tool1>...</tool1>
                    # re.DOTALL allows . to match newlines
                    if("<tool0>"in tool_resp):
                        match = tool_resp.split("<tool0>")[1].split("</tool0>")[0].strip()
                        processed = eval(match,  eval_env)
                        # 【Crucial update】Update environment variables so subsequent tools can use the processed image
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                     
                    if("<tool1>"in tool_resp):
                        match = tool_resp.split("<tool1>")[1].split("</tool1>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                    
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

                    if("<tool2>"in tool_resp):
                        match = tool_resp.split("<tool2>")[1].split("</tool2>")[0].strip()
                        processed = eval(match,  eval_env)
                        eval_env["image_np_small"] = processed
                        eval_env["image"] = processed
                 
                 
                        
                    if isinstance(processed, np.ndarray):
                        outputimg_np = processed
                        # ---------------------------
                        # Saving Processed Image to Custom Path
                        # ---------------------------
                        save_dir = self.processed_image_dir
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{index}_processed.png")
                        cv2.imwrite(save_path, processed)
    
                        # Create a new figure to prevent overwriting previous ones
                        # It's better to create independent objects for each display in multithreading
                        plt.figure(figsize=(10, 5))
                        
                        # Display original image (assume image is BGR, need to convert to RGB)
                        plt.subplot(1, 2, 1)
                        # Note: image variable is taken from context, might be image_np_small if resized above
                        img_to_show = image if 'image' in locals() else image_np_small
                        if img_to_show is not None:
                            plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
                        plt.title(f"Original Index: {index}")
                        plt.axis('off')

                        # Display processed image
                        plt.subplot(1, 2, 2)
                        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                        plt.title(f"Processed Index: {index}")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.show() 
                    else:
                        print(f"No valid tool tag found in response. Response was: {tool_resp[:100]}...")
                              
            except Exception as e:
                print(f"Tool error: {e}")
                if 'tool_resp' in locals():
                    print(f"Full response: {tool_resp}")
            
        # 3. Final inference
        try:
            if classify in [2,6]:
                final_img_np = outputimg_np
            else:
                final_img_np = resize_to_max80(outputimg_np,max_size=80)
            _, buffer = cv2.imencode(".jpg", final_img_np)
            final_b64 = base64.b64encode(buffer).decode("utf-8")

            
            # Simplify complex list structure to string for OpenAI
            final_prompt_template = prompt_map.get(classify, class8prompt)
            #
            answerformat=" Just give me the option.no matter what the final answer is,you should check the options given in prompt!!! Attention: Don't give the number or answer directly, you are doing a choice question!only answer A B C D,chose the option which matches your option. you should give answer between  <answer></answer> dont't include other things like brackets just number or character"
            option_check="Attention:Check carefully what every option is !Chose the exact option that matches your analysis,don't give the wrong option!!!"
            final_prompt = auxline +cotprompt++prompt+ final_prompt_template+answerformat
            
            resp2 = call_openai_api(self.api_key, self.base_url, self.model, final_prompt, "dummy_path_for_log", image=final_b64)
            try:
                os.makedirs(os.path.dirname(self.output_log_path), exist_ok=True)
                with open(self.output_log_path, "a", encoding="utf-8") as f:
                    f.write(f"=== Index {index} ===question{prompt}\n{resp2}\n\n")
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
    # Add outermost exception block to prevent a single thread from deadlocking the pool
    try:
        # Compatible with json dictionary or Pandas Series
        if isinstance(row, dict):
             # JSON mode logic handling
             image_path = row.get("image_path")
             prompt = row.get("prompt")

        else:
             # CSV logic handling
             if "Image" in row:
                 # Fetch Image column data (dictionary {'bytes': b'...'})
                 image_data = row["Image"]
                 
                 image_bytes = None
                 # If it is a dictionary structure
                 if isinstance(image_data, dict) and 'bytes' in image_data:
                     image_bytes = image_data['bytes']
                 # If it is directly bytes buffer
                 elif isinstance(image_data, bytes):
                     image_bytes = image_data
                 
                 if image_bytes:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    image_path = image_cv
                 else:
                    print(f"Warning: No byte data found in 'Image' column for row {idx}")
                    image_path = None
             else:
                 # Compatible with former CSV logic
                 image_path = row.get("image_path")

             # Fetch Question text
             prompt = row.get("Question", "")
             
        result = solver.solve(image_path, prompt ,idx)
        return idx, result
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, -1

# ---------------------------
# PATH UPDATE REQUIRED IN THIS FUNCTION:
# Change img_base_dir to your local image dataset folder.
# ---------------------------
def run_parallel(api_keys:list, base_url:str, model:str, output_log_path:str, output_txt: str, output_json: str = "model.json", input_file=None, max_workers: int = 5, df=None, img_base_dir="YOUR_IMAGES_DIR", processed_image_dir="YOUR_PROCESSED_IMAGES_DIR"):
    tasks = []
    
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    open(output_log_path, "w", encoding="utf-8")
    
    solvers=[]
    for key in api_keys:
        s = OpenAIAPISolver(api_key=key, base_url=base_url, model=model, output_log_path=output_log_path, processed_image_dir=processed_image_dir)
        solvers.append(s)

    # Check if input is JSON or CSV based on extension or explicit argument
    if df is None:
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Base logic for JSON using updated generic dir
            for item in data:
                # Extract index from image_name (e.g., "0.jpg" -> 0)
                image_name = item.get("image_name", "")
                try:
                    idx = int(image_name.split('.')[0])
                except ValueError:
                    continue # Skip if no valid index
                
                # Construct Image Path
                image_path = os.path.join(img_base_dir, image_name)
                
                # Construct Prompt
                question = item.get("Question", "")
                options = item.get("option", "")
                # Format: Question + Newline + Options
                full_prompt = f"{question}\n{options}"
                
                # Prepare row dictionary for solver
                row = {
                    "image_path": image_path,
                    "prompt": full_prompt
                }
                
                # Filter for testing specific index if needed
                if  idx<=5:
                    solver=solvers[idx%len(solvers)]
                    tasks.append((solver, idx, row))
                
        else:
            # CSV Logic
            df = pd.read_csv(input_file)
            csv_dir = os.path.dirname(os.path.abspath(input_file))
            df["image_path"] = df["image_path"].apply(lambda x: os.path.join(csv_dir, x))
            
            # Prepare tasks from DataFrame
            for idx, row in df.iterrows():
                solver=solvers[idx%len(solvers)]
                tasks.append((solver, idx, row))
   
    else:
        for idx,row in df.iterrows():
            if idx<90:
                solver=solvers[idx%len(solvers)]
                tasks.append((solver, idx, row))

    # Run Executor
    results = {}
    
    # If tasks are empty (e.g. filter applied), handle gracefully
    if not tasks:
        print("No tasks to process.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for future in tqdm(concurrent.futures.as_completed([executor.submit(process_single_row, t) for t in tasks]), total=len(tasks)):
            try:
                idx, result = future.result(timeout=240) # 4 minutes timeout per individual task deadlock
                results[idx] = result
            except concurrent.futures.TimeoutError:
                print("Task timed out.")
            except Exception as e:
                print(f"Task failed: {e}")

    sorted_indices = sorted(results.keys())
    
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "a") as f:
        for idx in sorted_indices: f.write(f"{idx} {results[idx]}\n")
        
    with open(output_json, "w") as f: json.dump(solvers[0].model_info(), f, indent=2)

# %%
# Execution example context handling
if __name__ == "__main__":
    pass
    # ---------------------------
    # PATH UPDATE REQUIRED:
    # 1. Provide exact variables for user testing
    # 2. Update placeholders with your local file structure
    # ---------------------------
    # api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"]
    # input_json_file = "YOUR_INPUT_JSON_PATH/test.json"
    # output_txt = "YOUR_OUTPUT_TXT_PATH/test_result2.txt"
    # output_log_path = "YOUR_OUTPUT_LOG_PATH/cot_trace.txt"
    # processed_image_dir = "YOUR_PROCESSED_IMAGES_DIR"
    # img_base_dir = "YOUR_IMAGES_DIR"
    
    # # Call run_parallel with input_file pointing to JSON
    # run_parallel(api_keys=api_keys, 
    #              base_url="YOUR_API_BASE_URL", 
    #              model="gemini-3.1-pro-preview-cli", 
    #              output_log_path=output_log_path,
    #              output_txt=output_txt, 
    #              input_file=input_json_file, 
    #              max_workers=10,
    #              img_base_dir=img_base_dir,
    #              processed_image_dir=processed_image_dir)