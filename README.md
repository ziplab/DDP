<p align="center">
  <h1 align="center">Less Detail, Better Answers: Degradation-Driven Prompting for VQA</h1>
  <p align="center">
  <a href="https://github.com/hhx-jpg">Haoxuan Han</a>
    ·
    <a href="https://lhmd.top">Weijie Wang</a>
    ·
    <a href="https://steve-zeyu-zhang.github.io">Zeyu Zhang</a>
    ·
    <a href="https://hexy.tech/">Yefei He</a>
    ·
    <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a>
  </p>
  <h3 align="center"><a href="#">Paper</a> | <a href="#">Project Page</a> | <a href="#">Code</a>  </h3>
  <div align="center"></div>
</p>

Recent advancements in Vision-Language Models (VLMs) have significantly pushed the boundaries of Visual Question Answering (VQA). However, high-resolution details can sometimes become noise that leads to hallucinations or reasoning errors. We propose Degradation-Driven Prompting (DDP), a novel framework that improves VQA performance by strategically reducing image fidelity and utilizing an agentic tool-use pipeline to force models to focus on essential structural information.

## Method

<strong>Overview of Degradation-Driven Prompting (DDP)</strong>. Given an image and a question as input, our DDP framework introduces a "divide-and-conquer" workflow consisting of three stages:

 1) The <strong>Classifier</strong> categorizes the image type and visual task. 

2) The <strong>Tool-manager</strong> invokes specialized visual tools (e.g., draw rectangle, crop, blur masks, grid auxlines) to highlight suspicious regions and intentionally degrade distracting textures. 

3) The <strong>Critic</strong> synthesizes these visual cues, corrects initial misconceptions bridging the perception-logic gap, and provides the final reasoned answer. This active agentic perception approach allows VLMs to bypass deceiving high-frequency textures and achieve superior reasoning accuracy on challenging visual benchmarks.

## Project Structure

```
DDP/
├── asset/                  # Test data and images
│   ├── Data/               # Task datasets
│   │   ├── task1/
│   │   └── task2/
│   ├── TeT benchmark/  
│   ├── V_star bench/ 
│   
├── src/                    # Source code
│   ├── task1.py            # Task 1 solver (Optical illusions / Visual perception)
│   ├── task2.py            # Task 2 solver (Counting, Color blindness, Geometry, Real scenes, etc.)
│   ├── helper.py           # Base solver definitions
│   ├── mcp_server.py       # MCP Server (Optional, use with an MCP Client)
│  
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```
This is a professional and structured addition for your `README.md`. I have organized it to clearly distinguish between the two tasks while highlighting the strict "No Training" and "Institutional Email" requirements.

---

##  CVPR 2026 DataCV Challenge

This project is part of the **5th DataCV Challenge**, held in conjunction with the **CVPR 2026 DataCV Workshop**. The challenge focuses on the data-centric evaluation of Vision-Language Models (VLMs) under visual illusions and perceptual anomalies.

###  Challenge Overview
The core objective is to improve VLM robustness **without any model training or fine-tuning**. Participants must rely on prompting, in-context learning (ICL), and inference-time strategies using frozen, off-the-shelf models.

---

###  Task I: Classic Illusion Understanding
**Goal:** Design a strategy to enable a fixed VLM to answer binary (**Yes/No**) questions about classic optical illusions.

* **Input:** Illusion image + Binary question.
* **Core Constraint:**  **Perception-focused.** Submissions **must not** use measurement- or computation-based pipelines (e.g., explicit length/angle estimation, ruler-based quantification, or pixel-level statistics).
* **Allowed:** Basic adjustments like resizing or standard normalization that do not produce quantitative measurements.

---

### Task II: Real-world Visual Illusions and Anomalies
**Goal:** Design a strategy for a VLM to answer **multiple-choice questions** (A, B, C, D) based on real-world visual anomalies.

* **Input:** Image + Multiple-choice prompt.
* **Strategy:** Any form of prompting or inference-time strategy is allowed, provided the model remains frozen.

---

###  Common Constraints & Rules
To ensure a valid entry, all participants must adhere to the following:

1.  **No Training/Fine-tuning:** Strictly no gradient updates or weight changes are permitted ($0$ parameters updated).
2.  **Model Selection:** Only off-the-shelf, publicly released models (e.g., GPT-4, Claude, Qwen, LLaVA) are allowed.
3.  **Inference Only:** Only zero-shot or few-shot inference-time methods (Prompting, ICL) are permitted.


---
## Quick Start: Run Task Code Directly

`task1.py` and `task2.py` are **complete scripts that can run independently** without the MCP Server. They invoke the LLM API via an **OpenAI-compatible interface** to implement a three-stage pipeline: "Classification → Tool Usage → Final Reasoning".

### Configurations Required Before Running

At the bottom of `src/task1.py` and `src/task2.py` in the `__main__` area, you need to fill in the following information yourself:

#### 1. API Key

```python
# Supports multiple keys for concurrency, fill in your own keys
my_api_keys = ["sk-your-key-1", "sk-your-key-2", ...]
```

#### 2. API Base URL

```python
# Must be an OpenAI-compatible interface address (ending with /v1)
# The code will automatically append /chat/completions
base_url = "https://your-api-provider.com/v1"
```

> **Note**: The current code uses the **OpenAI Chat Completion interface format** (`/v1/chat/completions`), including the Vision multimodal message structure. If you are using official SDKs from other models like Gemini or Claude, you need to call them through an OpenAI-compatible relay service, or modify the `call_openai_api()` function yourself.

#### 3. File Paths

```python
# Input data path (CSV or JSON)
input_csv = "../asset/test.csv"       # task1
input_json_file = "path/to/test.json" # task2

# Output result path
output_txt = "path/to/result.txt"
```

#### 4. Model Name

```python
model_name = "gemini-3.1-pro-preview"  # Fill in according to your API provider's supported models
```

### Run

```bash
cd src
python task1.py   # Run Task 1
python task2.py   # Run Task 2
```

### Differences between Task 1 and Task 2

| | Task 1 | Task 2 |
|---|--------|--------|
| **Task Categories** | 4 types: color, size, line, other | 8 types: counting, spot-the-difference, color blindness, dynamic illusions, geometry, real scenes, size, other |
| **Answer Format** | Binary judgment 0/1 (Yes/No) | Multiple choice A/B/C/D |
| **Image Processing Tools** | whitemask, gridmask, crop, binary masks, edge/contrast enhance, equal-spacing lines | whitemask, gridmask, crop, draw_rectangle, reversed_blur_mask, enhance_contrast |
| **Data Format** | CSV (`image_path`, `prompt`) | JSON or Parquet (contains embedded images) |

### Workflow

Both Tasks share the same three-stage **DDP** architecture:

```text
Input Image + Question
      │
      ▼
┌─────────────┐
│1. Classifier│  Categorizes the image type and visual task (e.g., real picture, optical illusion).
└──────┬──────┘
       ▼
┌─────────────┐
│2. Tool-     │  Invokes specialized visual tools based on the category (e.g., draw rectangle, crop)
│   manager   │  → Code executes the tool function to degenerate textures and highlight explicit geometry.
└──────┬──────┘
       ▼
┌─────────────┐
│3. Critic    │  Synthesizes visual cues, detects mismatches, corrects misconceptions, and provides the
│             │  final reasoned answer bridging the perception-logic gap.
└─────────────┘
```

### Proxy Settings (Optional)

`task1.py` contains a proxy configuration. If your network environment does not require a proxy, please comment out or remove the `proxies` parameter in `call_openai_api()`:

```python
# Remove or comment out this section
proxies={
    "http": "http://YOUR_PROXY_IP:PORT",
    "https": "http://YOUR_PROXY_IP:PORT",
}
```

---

## MCP Server (Optional)

> **Prerequisite**: You need a client that supports the MCP protocol (such as Claude Desktop, Claude Code, or another MCP Client).

`src/mcp_server.py` extracts **all 12 image processing tools** from task1 and task2 into independent MCP services, exposing them to the MCP Client through the stdio transport protocol.

### What is this for?

The MCP Server allows you to call these image processing tools **interactively** within an MCP Client. When combined with the prompts defined in the code (such as `toolusage_1`, `toolusage_2`, `toolusage_3`, etc.), you can manually debug and analyze individual images without running the entire batch pipeline.

**Typical Use Cases**:
- Execute a toolchain step-by-step on a single image in Claude Desktop to observe the processing effect at each step
- Debug the tool selection strategy and parameters a specific category
- Expose tool capabilities to other MCP-compatible AI Agents

### Tool List

| Tool | Description |
|------|-------------|
| `resize_image` | Scale the image to the specified maximum dimension |
| `whitemask` | Keep the specified circular area, turn the rest white |
| `gridmask` | Draw vertical/horizontal/polar coordinate grid lines |
| `crop_image` | Crop to one or more bounding boxes |
| `near_white_to_binary` | Binarization of near-white pixels |
| `near_red_to_binary` | Extract near-red pixels as a black and white image |
| `laplacian_edge_enhance` | Laplacian edge enhancement |
| `enhance_luminance_contrast` | Strong luminance and contrast enhancement (CLAHE + S-curve) |
| `draw_equal_spacing_lines` | Draw three equidistant vertical guide lines |
| `reversed_blur_mask` | Selective blur (inside/outside a circle) |
| `enhance_contrast` | CLAHE contrast enhancement |
| `draw_rectangle` | Draw a highlighted rectangular box |

### Start MCP Server

```bash
python src/mcp_server.py
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "image-tools": {
      "command": "python",
      "args": ["D:/DDP/src/mcp_server.py"]
    }
  }
}
```

### Claude Code Configuration

Add to `.mcp.json` in the project root directory:

```json
{
  "mcpServers": {
    "image-tools": {
      "command": "python",
      "args": ["src/mcp_server.py"]
    }
  }
}
```

### Important Note

The MCP Server only provides the **image processing tools** themselves. To achieve the exact same results as the complete pipeline in task1/task2, you need to use these tools in your MCP Client alongside the classification prompts and tool invocation prompts defined in the code. These prompts are defined in the `solve()` methods of `task1.py` / `task2.py` (e.g., `classifyprompt`, `toolusage_1`, `class1prompt`, etc.).
