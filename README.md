# Motion Control Studio

A local FastAPI web app for transferring a reference video's body motion to a character image.
This is an **open-source approximation, not Kling**. It does not use Kling's models, API, or
proprietary workflow. The Full backend submits the Wan workflow that you provide to your local
ComfyUI instance; Ollama is only used to inspect the character/reference and draft a prompt.

## What it does

- Accepts a PNG/JPG character image and an MP4/MOV reference (up to 10 seconds and 50 MB).
- Samples five frames with FFmpeg and asks a local Ollama vision model to describe the subject
  and motion. You can edit the resulting prompt before generation.
- **Full:** submits your API-format workflow to a local ComfyUI server, polls its history, and
  downloads the generated MP4. You must supply a compatible Wan 2.2 Animate workflow and
  configure its node IDs and input names.
- **Lite:** tracks pose with MediaPipe and warps the character image into a 2D
  **low quality preview**. It is not generative video, can look rough, and does not reconstruct
  hidden body parts or reliably handle occlusion and complex turns. It runs without a GPU.
- Optionally replaces the generated result's audio with the reference video's audio.
- Files, model calls, and output stay on this computer when the services are configured at their
  local defaults. Uploaded files are stored under `data/uploads/` and are not automatically
  deleted; remove that folder when you no longer need the files.

## Requirements

- Python 3.11 or 3.12 (MediaPipe's Python wheel availability varies by operating system).
- FFmpeg **including `ffprobe`**, available on `PATH`.
- Ollama installed and running locally, plus a vision model such as `qwen2.5vl`.
- For Full mode: local ComfyUI, the Wan 2.2 Animate custom nodes/models required by your
  workflow, sufficient GPU memory for that workflow, and a workflow exported in API format.
- For Lite mode: MediaPipe, OpenCV, and NumPy (installed with this project's dependencies).

Wan 2.2 generation is demanding. Required VRAM depends on the exact model, resolution,
precision, frame count, and workflow; consult the model/workflow documentation and start with
conservative settings. Motion Control Studio cannot guarantee that a particular GPU can run it.

## Setup

### Windows (PowerShell)

Install Python 3.11 or 3.12, FFmpeg (for example `winget install Gyan.FFmpeg`), and Ollama.
Restart the terminal after FFmpeg installation so its `bin` directory is on `PATH`.

```powershell
cd D:\MotionControlStudio
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen2.5vl
ollama serve
```

In a second PowerShell window:

```powershell
cd D:\MotionControlStudio
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

### macOS

Install Python 3.11/3.12, FFmpeg and Ollama (for example `brew install python@3.12 ffmpeg ollama`).

```bash
cd /path/to/MotionControlStudio
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5vl
ollama serve
```

In another terminal, activate the environment and run:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

### Linux

Install Python 3.11/3.12, the matching Python venv package, FFmpeg, and Ollama using your
distribution's package instructions. On Ubuntu/Debian, FFmpeg can be installed with
`sudo apt install ffmpeg`. Then:

```bash
cd /path/to/MotionControlStudio
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5vl
ollama serve
```

In another terminal, activate the environment and run
`python -m uvicorn main:app --host 127.0.0.1 --port 8000`, then open
<http://127.0.0.1:8000>.

If your `qwen2.5vl` installation uses a tag such as `qwen2.5vl:7b`, set `VISION_MODEL` to the
exact tag in `.env`. The health banner shows which local services/models are not available and
suggests the relevant command or setting.

## Configure Full mode

1. Install and start ComfyUI on this computer. Make sure the workflow's required Wan 2.2 Animate
   nodes and models are installed and working in ComfyUI.
2. Open the workflow in the ComfyUI web UI. Enable **Dev mode options** in Settings, then use
   **Save (API Format)** (not the regular workflow save) to export the API prompt JSON.
3. Save the exported JSON as `workflows/wan_animate_api.json`, replacing the included empty
   placeholder. This project intentionally does not invent or redistribute a Wan workflow:
   exported node IDs and custom-node inputs vary by installation.
4. In the exported JSON, identify the image loader, video input, and text/prompt nodes. Add their
   IDs and the exact input field names to `.env`, for example:

   ```dotenv
   IMAGE_NODE_ID=42
   IMAGE_INPUT_NAME=image
   VIDEO_NODE_ID=57
   VIDEO_INPUT_NAME=video
   PROMPT_NODE_ID=61
   PROMPT_INPUT_NAME=text
   ```

   These are examples only; use your own workflow's node IDs/keys. If your workflow has an
   orientation control node, set `ORIENTATION_NODE_ID` and `ORIENTATION_INPUT_NAME` as well.
   Otherwise, the selected orientation is included as a short instruction in the prompt.
5. Set `COMFYUI_URL` and `COMFYUI_INPUT_DIR` to match the local ComfyUI server and its `input`
   folder, then restart Motion Control Studio. For example:

   ```dotenv
   COMFYUI_URL=http://127.0.0.1:8188
   COMFYUI_INPUT_DIR=C:\ComfyUI\input
   ```

   The input folder must be writable by this app. ComfyUI must be reachable from this computer.
6. Check the **Wan workflow** and **ComfyUI** health items before choosing Full mode. The app
   copies uploaded references into ComfyUI's input folder and downloads MP4 outputs from the
   ComfyUI `/view` endpoint.

The workflow is user-provided and is not bundled, so Full mode is not ready until you export and
configure it. Without a configured Full workflow, use Lite for a local low quality preview.

## Settings

Settings can be set in `.env` (copied from `.env.example`) or environment variables:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama base URL |
| `VISION_MODEL` | `qwen2.5vl` | Ollama vision model/tag |
| `VISION_TIMEOUT_SECONDS` | `600` | Vision analysis timeout (raise for slow CPUs) |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | Local ComfyUI base URL |
| `COMFYUI_INPUT_DIR` | Auto-detect `~/ComfyUI/input` | Writable ComfyUI input folder |
| `WORKFLOW_PATH` | `workflows/wan_animate_api.json` | API-format workflow JSON |
| `IMAGE_NODE_ID`, `VIDEO_NODE_ID`, `PROMPT_NODE_ID` | `10`, `11`, `12` | Workflow node IDs |
| `IMAGE_INPUT_NAME`, `VIDEO_INPUT_NAME`, `PROMPT_INPUT_NAME` | `image`, `video`, `text` | Input keys at those nodes |
| `ORIENTATION_NODE_ID` | Empty | Optional workflow orientation node |
| `ORIENTATION_INPUT_NAME` | `orientation` | Optional orientation input key |
| `COMFY_TIMEOUT_SECONDS` | `600` | Maximum time to poll a ComfyUI job |
| `DATA_DIR` | `data` | Local upload/result storage |

## Run the fake-server tests

The tests use local fake Ollama and ComfyUI HTTP servers; they do not require a real model,
ComfyUI server, GPU, or FFmpeg binary:

```bash
python -m pytest -q
```

The tests cover vision response parsing and person rejection, prompt/image/frame transfer to
fake Ollama, ComfyUI workflow input substitution and MP4 download, ComfyUI timeout/OOM errors,
and upload validation. Lite's MediaPipe/OpenCV render path is exercised only when running Lite
with actual reference media; output quality depends on the input image and pose tracking.

## Troubleshooting

- **Ollama unavailable:** start it with `ollama serve`; verify `OLLAMA_URL`. Pull the model with
  `ollama pull qwen2.5vl`.
- **Vision model missing:** choose a locally installed vision-capable Ollama model in `.env`.
- **FFmpeg unavailable:** install FFmpeg (including `ffprobe`), add its executable directory to
  `PATH`, and restart the app.
- **ComfyUI unavailable:** start ComfyUI and verify `COMFYUI_URL`; Full mode needs it, Lite does
  not.
- **Wan workflow missing/invalid:** export **Save (API Format)**, replace the placeholder JSON,
  and set the node IDs/input keys to those in your exported workflow.
- **ComfyUI out of memory:** lower frame count, resolution, or batch size in your workflow and
  check your GPU's available memory.
- **Lite dependencies unavailable:** use Python 3.11/3.12 and reinstall with
  `pip install -r requirements.txt`.
- **Lite cannot segment the character:** use a transparent PNG or a clearly contrasting,
  uncluttered background.

## Privacy and network binding

Run with the documented `127.0.0.1` host binding to keep the web app local. Do not expose it to
an untrusted network: this development app does not implement user authentication. Ollama and
ComfyUI should also remain on trusted local endpoints.
