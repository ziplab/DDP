# DDP - Visual Analysis with LLM Tool Use

A framework that solves visual perception tasks through an LLM API + image processing toolchain, covering various tasks such as optical illusion analysis, color comparison, size measurement, line geometry judgment, counting, spot-the-difference, and color bindness test recognition.

## Project Structure

```
DDP/
├── asset/                  # Test data and images
│   ├── Data/               # Task datasets
│   │   ├── task1/
│   │   └── task2/
│   ├── direct_attributes/  # Attribute images
│   ├── relative_position/  # Relative position images
│   └── test.csv
├── src/                    # Source code
│   ├── task1.py            # Task 1 solver (Optical illusions / Visual perception)
│   ├── task2.py            # Task 2 solver (Counting, Color blindness, Geometry, Real scenes, etc.)
│   ├── helper.py           # Base solver definitions
│   ├── mcp_server.py       # MCP Server (Optional, use with an MCP Client)
│   └── task1_temp.py       # Experimental code
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

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

Both Tasks share the same three-stage architecture:

```
Input Image + Question
      │
      ▼
┌─────────────┐
│ 1. Classify │  LLM determines which category the question belongs to
└──────┬──────┘
       ▼
┌─────────────┐
│ 2. Tool Use │  LLM selects image processing tools based on the category and provides parameters
│             │  → Code executes the tool function and generates the processed image
└──────┬──────┘
       ▼
┌─────────────┐
│ 3. Reason   │  LLM provides the final answer based on the processed image + a dedicated prompt
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
