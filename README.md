# RunPod ComfyUI WAN 2.2 frame-to-frame setup

Startup-time installer for `WAN2.2_base_Q8_max_realism_20H20L.json`. It is
designed for RunPod pods that use Container Disk only. The current lean profile
downloads only the assets needed by the frame-to-frame branch. Q8 and shared
assets are made ready first; matching FP8 alternatives continue downloading
after ComfyUI starts.

This repository is the mutable WAN-specific setup layer. It reuses the shared
SCAIL/WAN wrapper images that provide the pinned RunPod CUDA 12.8 base,
SageAttention, baked external custom nodes and Python packages, FileBrowser,
Jupyter, and the original `/start.sh` handoff.

## RunPod template

Choose the wrapper matching the GPU generation:

- RTX 30-series: `nightfall93/comfyui-scail-wrapper:sage2-ampere`
- RTX 40-series: `nightfall93/comfyui-scail-wrapper:sage2-ada`
- RTX 50-series: `nightfall93/comfyui-scail-wrapper:sage2-blackwell`

Use at least **100 GB of Container Disk**. The public downloads are roughly
68 GB before the image, Python packages, caches, inputs, and generated output.
No Volume Disk or Network Volume is required, but deleting or resetting the
container also deletes all completed and partial downloads.

Expose these HTTP ports:

- `8188` for ComfyUI
- `8080` for FileBrowser
- `8888` for JupyterLab

After this repository is published, set:

```text
SETUP_SCRIPT_URL=https://raw.githubusercontent.com/Nightfall93/runpod-comfyui-wan22/main/wan22_download_setup.sh
```

The script derives the workflow URL from `SETUP_SCRIPT_URL`. Set
`WAN22_WORKFLOW_URL` only when the workflow is hosted somewhere else.
It also installs the repository's model-pair switch and LightX loader nodes. Set
`WAN22_SWITCH_NODE_URL` only when that file is hosted somewhere else.

Other supported environment variables:

- `FILEBROWSER_USERNAME`
- `FILEBROWSER_PASSWORD`
- `JUPYTER_PASSWORD`
- `HF_TOKEN` (optional; also authorizes private Hugging Face download URLs)
- `NTFY_TOPIC` (optional)
- `NTFY_SERVER_URL` (optional; defaults to `https://ntfy.sh`)
- `NTFY_TOKEN` (optional)
- `WAN22_DOWNLOAD_JOBS` (optional; defaults to `2`, allowed range `1`-`4`)
- `WAN22_FP8_BACKGROUND` (optional; defaults to `1`; set `0` to block startup
  until FP8 is also ready)
- `WAN22_REFRESH_WORKFLOW` (optional; set `1` to replace the installed workflow
  with the repository copy on that startup)

## Downloaded model assets

Diffusion models:

- WAN 2.2 I2V high-noise Q8_0
- WAN 2.2 I2V low-noise Q8_0
- WAN 2.2 I2V high-noise FP8 scaled safetensor
- WAN 2.2 I2V low-noise FP8 scaled safetensor

Frame-to-frame text encoder and VAE:

- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `wan_2.1_vae.safetensors`

LoRAs:

- LightX2V WAN 2.2 I2V high and low

The frame-to-frame branch has one **MODEL FORMAT** checkbox. Leave it off for
the Q8 GGUF pair or turn it on for the FP8 safetensor pair after its background
download reports `state=ready`. The switch owns the loaders and loads only the
selected pair, so missing in-progress FP8 files do not prevent Q8 prompt
validation and both formats are not loaded into memory at the same time.

The adjacent **LIGHTX 4-STEP - FULL KEY SUPPORT** node controls the two 1022
acceleration LoRAs together. Its WAN-specific loader performs the same
`.diff_m` to `.modulation.diff` normalization as LightX2V's official
WanVideoWrapper workflow before calling ComfyUI's patcher. This prevents the
per-block modulation tensors from being reported and ignored as unloaded LoRA
keys. The node is off by default and exposes separate high- and low-noise
strengths for controlled comparisons.

On startup, an untouched copy of the previously published workflow is upgraded
to this corrected graph automatically. A workflow edited in ComfyUI is
preserved; set `WAN22_REFRESH_WORKFLOW=1` for one startup if you intentionally
want to replace your customized copy.

Background FP8 progress is available at:

- `/workspace/runpod-slim/wan22-fp8-download.status`
- `/workspace/runpod-slim/wan22-fp8-download.log`

If FP8 is selected too early, the switch produces a clear error and Q8 remains
usable. No ComfyUI restart is required after the final `.part` files are moved
into place.

The full canvas still contains other selectable branches, but their model assets
are intentionally not downloaded by this lean profile.

Sample images and videos embedded as widget selections are not dependencies.
Upload your own inputs after ComfyUI starts.

## Custom nodes

Updated wrapper images bake these external installations at pinned commits.
This setup script retains the same pinned installers as a fallback for older
or third-party images:

- ComfyUI-GGUF
- ComfyUI-Easy-Use
- rgthree-comfy
- comfyui-vrgamedevgirl
- ComfyUI_essentials
- ComfyUI-Frame-Interpolation
- ComfyUI_UltimateSDUpscale
- wlsh_nodes
- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- RES4LYF (`res_2s` sampler and `bong_tangent` scheduler)
- WAN 2.2 model-pair switch and full-key LightX loader (installed from this
  repository)

An installation already present in the RunPod base image is preserved.

## Startup behavior

The installer performs a real CUDA tensor preflight before consuming bandwidth,
installs missing custom nodes, probes download sizes, resumes `.part` files,
and reconnects persistently slow transfers. Up to two downloads run at once by
default. It first completes the Q8 pair, text encoder, VAE, and LoRAs, then
validates and installs the workflow and returns control to the wrapper. The
wrapper enables SageAttention and starts the original RunPod services while a
supervised background worker downloads the two FP8 models concurrently.

Background failure is written to the status file and sent through ntfy when it
is configured. Partial FP8 files are preserved for the next startup. Setting
`WAN22_FP8_BACKGROUND=0` restores blocking behavior while retaining concurrent
downloads.

Completed files are skipped on later starts while the same container filesystem
still exists.
