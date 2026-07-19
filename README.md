# RunPod ComfyUI WAN 2.2 frame-to-frame setup

Startup-time installer for `WAN2.2_base_Q8_max_realism_20H20L.json`. It is
designed for RunPod pods that use Container Disk only. The current lean profile
downloads only the assets needed by the frame-to-frame branch, plus matching
FP8 diffusion-model alternatives, before ComfyUI starts.

This repository is the mutable setup layer. It reuses the existing wrapper
images that provide the pinned RunPod CUDA 12.8 base, SageAttention, CUDA
preflight, FileBrowser, Jupyter, and the original `/start.sh` handoff.

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
It also installs the repository's small lazy model-pair switch node. Set
`WAN22_SWITCH_NODE_URL` only when that file is hosted somewhere else.

Other supported environment variables:

- `FILEBROWSER_USERNAME`
- `FILEBROWSER_PASSWORD`
- `JUPYTER_PASSWORD`
- `HF_TOKEN` (optional; also authorizes private Hugging Face download URLs)
- `NTFY_TOPIC` (optional)
- `NTFY_SERVER_URL` (optional; defaults to `https://ntfy.sh`)
- `NTFY_TOKEN` (optional)

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
the Q8 GGUF pair or turn it on for the FP8 safetensor pair. The switch is lazy:
only the selected high- and low-noise loaders execute, so both pairs are not
loaded into memory at the same time.

The full canvas still contains other selectable branches, but their model assets
are intentionally not downloaded by this lean profile.

Sample images and videos embedded as widget selections are not dependencies.
Upload your own inputs after ComfyUI starts.

## Custom nodes

Missing installations are fetched at pinned commits:

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
- WAN 2.2 lazy model-pair switch (installed from this repository)

An installation already present in the RunPod base image is preserved.

## Startup behavior

The installer performs a real CUDA tensor preflight before consuming bandwidth,
installs missing custom nodes, probes download sizes, resumes `.part` files,
reconnects persistently slow transfers, validates and installs the workflow,
patches FileBrowser credentials, and returns control to the wrapper. The wrapper
then enables SageAttention when compatible and starts the original RunPod
services.

Completed files are skipped on later starts while the same container filesystem
still exists.
