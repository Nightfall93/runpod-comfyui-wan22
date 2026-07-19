#!/usr/bin/env bash
set -euo pipefail

COMFY="/workspace/runpod-slim/ComfyUI"
VENV="$COMFY/.venv-cu128"
DOWNLOAD_JOBS="${WAN22_DOWNLOAD_JOBS:-2}"
FP8_STATUS_FILE="/workspace/runpod-slim/wan22-fp8-download.status"
FP8_LOG_FILE="/workspace/runpod-slim/wan22-fp8-download.log"

if ! [[ "$DOWNLOAD_JOBS" =~ ^[1-4]$ ]]; then
  echo "WARNING: WAN22_DOWNLOAD_JOBS must be from 1 to 4; using 2."
  DOWNLOAD_JOBS=2
fi

echo "=== WAN 2.2 frame-to-frame setup starting ==="

notify_ntfy() {
  local title="$1"
  local priority="$2"
  local tags="$3"
  local message="$4"
  local server="${NTFY_SERVER_URL:-https://ntfy.sh}"
  local -a auth=()

  [ -n "${NTFY_TOPIC:-}" ] || return 0
  if [ -n "${NTFY_TOKEN:-}" ]; then
    auth=(-H "Authorization: Bearer ${NTFY_TOKEN}")
  fi

  if ! curl -fsS --max-time 10 --retry 2 \
    "${auth[@]}" \
    -H "Title: $title" \
    -H "Priority: $priority" \
    -H "Tags: $tags" \
    --data-binary "$message" \
    "${server%/}/${NTFY_TOPIC#/}" >/dev/null; then
    echo "WARNING: Could not send ntfy notification: $title"
  fi
}

echo "Running CUDA preflight before model setup..."
cuda_ready=0
for attempt in 1 2 3; do
  if cuda_details=$(timeout 20s python3 - <<'PY_CUDA' 2>&1
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")

device = torch.cuda.current_device()
probe = torch.zeros(1, device=device)
torch.cuda.synchronize(device)
major, minor = torch.cuda.get_device_capability(device)
print(
    f"{torch.cuda.get_device_name(device)} "
    f"(compute capability {major}.{minor}, PyTorch CUDA {torch.version.cuda})"
)
del probe
PY_CUDA
  ); then
    echo "CUDA preflight passed: $cuda_details"
    cuda_ready=1
    break
  fi

  echo "CUDA preflight attempt $attempt/3 failed:"
  printf '%s\n' "$cuda_details"
  if [ "$attempt" -lt 3 ]; then
    echo "Retrying CUDA preflight in 5 seconds..."
    sleep 5
  fi
done

if [ "$cuda_ready" -ne 1 ]; then
  echo "FATAL: CUDA compute is unavailable after 3 attempts."
  echo "Model setup was not started, so replace or reset this RunPod host."
  notify_ntfy \
    "RunPod CUDA failure" "urgent" "warning" \
    "CUDA compute failed its startup check. No models were downloaded; replace or reset this host."
  exit 1
fi

# IMPORTANT:
# comfyui-base normally copies ComfyUI into /workspace on first boot.
# If we create /workspace/runpod-slim/ComfyUI too early, we can break that copy.
# So this script first copies the baked ComfyUI if needed.
if [ ! -f "$COMFY/main.py" ]; then
  echo "ComfyUI not found in /workspace yet. Copying baked ComfyUI..."
  mkdir -p /workspace/runpod-slim
  cp -r /opt/comfyui-baked "$COMFY"
fi

# Create/activate the comfyui-base CUDA 12.8 venv if needed.
if [ ! -d "$VENV" ]; then
  echo "Creating ComfyUI venv..."
  cd "$COMFY"
  python3.12 -m venv --system-site-packages "$VENV"
  source "$VENV/bin/activate"
  python -m ensurepip
else
  source "$VENV/bin/activate"
fi

mkdir -p \
  "$COMFY/custom_nodes" \
  "$COMFY/models/diffusion_models/Wan2.2" \
  "$COMFY/models/loras/Wan2.2" \
  "$COMFY/models/text_encoders" \
  "$COMFY/models/vae" \
  "$COMFY/user/default/workflows"

if [ -n "${SETUP_SCRIPT_URL:-}" ]; then
  SCRIPT_BASE_URL="${SETUP_SCRIPT_URL%/*}"
else
  SCRIPT_BASE_URL=""
fi

format_eta() {
  local seconds="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((seconds / 3600))" "$(((seconds / 60) % 60))" "$((seconds % 60))"
}

remote_size() {
  local url="$1"
  local -a headers=()
  if [ -n "${HF_TOKEN:-}" ]; then
    headers=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  curl -fsSIL "${headers[@]}" "$url" | awk '
    BEGIN { IGNORECASE=1 }
    /^content-length:/ { size=$2 }
    END { gsub(/\r/, "", size); print size }
  ' || true
}

download_file() {
  local kind="$1"
  local url="$2"
  local out="$3"
  local expected_size="$4"
  local part="${out}.part"
  local curl_log="${part}.curl.log"
  local name started initial_bytes now bytes elapsed transferred average_speed
  local sample_started sample_elapsed sample_bytes current_speed
  local file_remaining total_remaining file_eta total_eta
  local status_printed=0 last_status_elapsed=0
  local last_observed_bytes stall_notified=0
  local slow_samples=0 reconnect_count=0 reconnect_requested=0
  local reconnect_limit_reported=0 curl_pid
  local min_speed=$((5 * 1024 * 1024))
  local -a curl_headers=()
  mkdir -p "$(dirname "$out")"
  name="$(basename "$out")"

  if [ -s "$out" ]; then
    printf 'Ready %-5s  %s (already downloaded)\n' "$kind" "$name"
    return 0
  fi

  if [ -n "${HF_TOKEN:-}" ]; then
    curl_headers=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi

  started=$(date +%s)
  initial_bytes=$(stat -c '%s' "$part" 2>/dev/null || echo 0)

  while true; do
    reconnect_requested=0
    slow_samples=0
    last_observed_bytes=$(stat -c '%s' "$part" 2>/dev/null || echo 0)
    sample_started=$(date +%s)

    curl -L --fail --silent --show-error --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --speed-limit 1024 --speed-time 15 \
      "${curl_headers[@]}" \
      --continue-at - -o "$part" "$url" >"$curl_log" 2>&1 &
    curl_pid=$!

    while kill -0 "$curl_pid" 2>/dev/null; do
      sleep 10
      kill -0 "$curl_pid" 2>/dev/null || break
      now=$(date +%s)
      bytes=$(stat -c '%s' "$part" 2>/dev/null || echo 0)
      sample_elapsed=$((now - sample_started))
      [ "$sample_elapsed" -lt 1 ] && sample_elapsed=1
      sample_bytes=$((bytes - last_observed_bytes))
      [ "$sample_bytes" -lt 0 ] && sample_bytes=0
      current_speed=$((sample_bytes / sample_elapsed))

      if [ "$bytes" -le "$last_observed_bytes" ]; then
        if [ "$stall_notified" -eq 0 ]; then
          echo "WARNING: $kind $name has not grown in the last 10 seconds."
          notify_ntfy \
            "RunPod download stalled" "urgent" "warning" \
            "$kind $name has not changed size for 10 seconds. Curl is still retrying."
          stall_notified=1
        fi
      else
        stall_notified=0
      fi

      elapsed=$((now - started))
      [ "$elapsed" -lt 1 ] && elapsed=1
      transferred=$((bytes - initial_bytes))
      average_speed=$((transferred / elapsed))

      if [ "$transferred" -gt 0 ] \
        && { [ "$status_printed" -eq 0 ] || [ $((elapsed - last_status_elapsed)) -ge 10 ]; }; then
        if [[ "$expected_size" =~ ^[0-9]+$ ]] \
          && [ "$expected_size" -gt "$bytes" ] \
          && [ "$average_speed" -gt 0 ]; then
          file_remaining=$((expected_size - bytes))
          file_eta=$(((file_remaining + average_speed - 1) / average_speed))
          printf 'Downloading %-5s  %s | %s/%s | current %s/s | average %s/s | ETA %s\n' \
            "$kind" "$name" \
            "$(numfmt --to=iec-i --suffix=B "$bytes")" \
            "$(numfmt --to=iec-i --suffix=B "$expected_size")" \
            "$(numfmt --to=iec-i --suffix=B "$current_speed")" \
            "$(numfmt --to=iec-i --suffix=B "$average_speed")" \
            "$(format_eta "$file_eta")"
        else
          printf 'Downloading %-5s  %s | current %s/s | average %s/s | ETA unavailable\n' \
            "$kind" "$name" \
            "$(numfmt --to=iec-i --suffix=B "$current_speed")" \
            "$(numfmt --to=iec-i --suffix=B "$average_speed")"
        fi
        status_printed=1
        last_status_elapsed=$elapsed
      fi

      if [ "$current_speed" -lt "$min_speed" ]; then
        slow_samples=$((slow_samples + 1))
      else
        slow_samples=0
      fi

      if [ "$slow_samples" -ge 3 ]; then
        if [ "$reconnect_count" -lt 3 ]; then
          reconnect_count=$((reconnect_count + 1))
          echo "WARNING: $kind $name stayed below 5MiB/s for 30 seconds; reconnecting ($reconnect_count/3)."
          notify_ntfy \
            "RunPod download reconnecting" "urgent" "warning" \
            "$kind $name stayed below 5MiB/s for 30 seconds. Reconnecting attempt $reconnect_count/3 and resuming the partial file."
          reconnect_requested=1
          kill "$curl_pid" 2>/dev/null || true
        elif [ "$reconnect_limit_reported" -eq 0 ]; then
          echo "WARNING: $kind $name is still below 5MiB/s, but the 3-reconnect limit was reached; continuing."
          reconnect_limit_reported=1
        fi
        slow_samples=0
      fi

      last_observed_bytes=$bytes
      sample_started=$now
      [ "$reconnect_requested" -eq 1 ] && break
    done

    if [ "$reconnect_requested" -eq 1 ]; then
      if wait "$curl_pid" 2>/dev/null; then
        # Curl completed between the last sample and the termination request.
        break
      fi
      sleep 2
      continue
    fi

    if ! wait "$curl_pid"; then
      echo "Download failed after retries; showing the last curl errors:"
      tail -n 20 "$curl_log" || true
      return 1
    fi
    break
  done

  rm -f "$curl_log"
  mv "$part" "$out"

  now=$(date +%s)
  elapsed=$((now - started))
  bytes=$(stat -c '%s' "$out" 2>/dev/null || echo 0)
  transferred=$((bytes - initial_bytes))
  [ "$elapsed" -lt 1 ] && elapsed=1
  average_speed=$((transferred / elapsed))
  if [ "$status_printed" -eq 0 ]; then
    printf 'Downloading %-5s  %s | completed before first ETA sample\n' "$kind" "$name"
  fi
  printf 'Downloaded  %-5s  %s | %s | average %s/s\n' \
    "$kind" "$name" "$(numfmt --to=iec-i --suffix=B "$bytes")" \
    "$(numfmt --to=iec-i --suffix=B "$average_speed")"
  notify_ntfy \
    "RunPod download complete" "default" "white_check_mark" \
    "$kind $name finished downloading."

}

download_group() {
  local label="$1"
  shift
  local -a indices=("$@")
  local index next=0 active=0 failures=0 pid
  local started finished elapsed initial_bytes=0 final_bytes=0 transferred average_speed
  local path part_size name

  [ "${#indices[@]}" -gt 0 ] || return 0
  echo "Starting $label downloads with up to $DOWNLOAD_JOBS concurrent transfers."
  started=$(date +%s)

  for index in "${indices[@]}"; do
    path="${DOWNLOAD_OUTPUTS[$index]}"
    if [ -s "$path" ]; then
      initial_bytes=$((initial_bytes + $(stat -c '%s' "$path" 2>/dev/null || echo 0)))
    else
      part_size=$(stat -c '%s' "${path}.part" 2>/dev/null || echo 0)
      initial_bytes=$((initial_bytes + part_size))
    fi
  done

  while [ "$next" -lt "${#indices[@]}" ] || [ "$active" -gt 0 ]; do
    while [ "$next" -lt "${#indices[@]}" ] && [ "$active" -lt "$DOWNLOAD_JOBS" ]; do
      index="${indices[$next]}"
      name="$(basename "${DOWNLOAD_OUTPUTS[$index]}")"
      (
        download_file "${DOWNLOAD_KINDS[$index]}" "${DOWNLOAD_URLS[$index]}" \
          "${DOWNLOAD_OUTPUTS[$index]}" "${DOWNLOAD_SIZES[$index]}"
      ) 2>&1 | sed -u "s/^/[$label][$name] /" &
      pid=$!
      echo "[$label] Started $name (worker PID $pid)."
      next=$((next + 1))
      active=$((active + 1))
    done

    if [ "$active" -gt 0 ]; then
      if wait -n; then
        :
      else
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done

  finished=$(date +%s)
  elapsed=$((finished - started))
  [ "$elapsed" -lt 1 ] && elapsed=1
  for index in "${indices[@]}"; do
    final_bytes=$((final_bytes + $(stat -c '%s' "${DOWNLOAD_OUTPUTS[$index]}" 2>/dev/null || echo 0)))
  done
  transferred=$((final_bytes - initial_bytes))
  [ "$transferred" -lt 0 ] && transferred=0
  average_speed=$((transferred / elapsed))

  if [ "$failures" -gt 0 ]; then
    echo "ERROR: $label finished with $failures failed download worker(s)."
    return 1
  fi

  echo "Completed $label downloads in $(format_eta "$elapsed") at aggregate average $(numfmt --to=iec-i --suffix=B "$average_speed")/s."
}

write_fp8_status() {
  local state="$1"
  local detail="$2"
  local temp="${FP8_STATUS_FILE}.tmp"
  {
    printf 'state=%s\n' "$state"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'detail=%s\n' "$detail"
    printf 'log=%s\n' "$FP8_LOG_FILE"
  } >"$temp"
  mv "$temp" "$FP8_STATUS_FILE"
}

start_fp8_downloads() {
  local index all_ready=1 background_pid

  for index in "${FP8_DOWNLOADS[@]}"; do
    if [ ! -s "${DOWNLOAD_OUTPUTS[$index]}" ]; then
      all_ready=0
      break
    fi
  done

  if [ "$all_ready" -eq 1 ]; then
    write_fp8_status "ready" "Both FP8 diffusion models are available."
    echo "Ready FP8 background assets (already downloaded)."
    return 0
  fi

  if [ "${WAN22_FP8_BACKGROUND:-1}" = "0" ]; then
    echo "WAN22_FP8_BACKGROUND=0; waiting for the FP8 pair before startup."
    if download_group "WAN22-FP8" "${FP8_DOWNLOADS[@]}"; then
      write_fp8_status "ready" "Both FP8 diffusion models are available."
      return 0
    fi
    write_fp8_status "failed" "One or more FP8 downloads failed."
    return 1
  fi

  mkdir -p "$(dirname "$FP8_STATUS_FILE")"
  : >"$FP8_LOG_FILE"
  write_fp8_status "downloading" \
    "FP8 downloads are running; keep MODEL FORMAT set to Q8 GGUF."

  (
    trap '' HUP
    if download_group "WAN22-FP8-BG" "${FP8_DOWNLOADS[@]}"; then
      write_fp8_status "ready" "Both FP8 diffusion models are available."
      echo "FP8 background downloads are ready; MODEL FORMAT can now be switched to FP8."
      notify_ntfy \
        "WAN 2.2 FP8 models ready" "high" "white_check_mark" \
        "Both WAN 2.2 FP8 diffusion models finished downloading. The MODEL FORMAT switch can now use FP8."
      exit 0
    fi

    write_fp8_status "failed" \
      "One or more FP8 downloads failed; partial files will resume next startup."
    echo "ERROR: FP8 background downloads failed; Q8 remains available."
    notify_ntfy \
      "WAN 2.2 FP8 download failed" "urgent" "warning" \
      "One or more FP8 background downloads failed. Q8 remains usable and partial FP8 files will resume next startup."
    exit 1
  ) > >(tee -a "$FP8_LOG_FILE") 2>&1 &
  background_pid=$!
  disown "$background_pid" 2>/dev/null || true
  echo "FP8 pair downloading in the background (PID $background_pid)."
  echo "FP8 status: $FP8_STATUS_FILE"
  echo "FP8 log:    $FP8_LOG_FILE"
}

install_pinned_node() {
  local repo="$1"
  local dir="$2"
  local commit="$3"
  local part="${dir}.wan22-part"

  if [ -e "$dir" ] || [ -L "$dir" ]; then
    echo "Custom node already exists, preserving it: $dir"
    return 0
  fi

  echo "Installing pinned custom node: $repo @ $commit"
  rm -rf "$part"
  git init "$part"
  git -C "$part" remote add origin "$repo"
  git -C "$part" fetch --depth 1 origin "$commit"
  git -C "$part" checkout --detach FETCH_HEAD

  if [ -f "$part/requirements.txt" ]; then
    echo "Installing requirements for: $(basename "$dir")"
    if [ -f /opt/comfyui-runtime-constraints.txt ]; then
      pip install -c /opt/comfyui-runtime-constraints.txt -r "$part/requirements.txt"
    else
      pip install -r "$part/requirements.txt"
    fi
  fi

  rm -rf "$part/.git"
  mv "$part" "$dir"
}

# Custom nodes used directly by the workflow, plus RES4LYF which registers the
# res_2s sampler and bong_tangent scheduler used by its image branches.
install_pinned_node "https://github.com/city96/ComfyUI-GGUF.git" \
  "$COMFY/custom_nodes/ComfyUI-GGUF" \
  "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
install_pinned_node "https://github.com/yolain/ComfyUI-Easy-Use.git" \
  "$COMFY/custom_nodes/ComfyUI-Easy-Use" \
  "54d080bf6a4f52da287e984f305243c10db097f5"
install_pinned_node "https://github.com/rgthree/rgthree-comfy.git" \
  "$COMFY/custom_nodes/rgthree-comfy" \
  "27b4f4cdcf3b127c29d5d8135ac1536ecbd4c383"
install_pinned_node "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git" \
  "$COMFY/custom_nodes/comfyui-vrgamedevgirl" \
  "930874103d9ab6b9bf98bc108b0483b4fb2ada4e"
install_pinned_node "https://github.com/cubiq/ComfyUI_essentials.git" \
  "$COMFY/custom_nodes/ComfyUI_essentials" \
  "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9"
install_pinned_node "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git" \
  "$COMFY/custom_nodes/ComfyUI-Frame-Interpolation" \
  "26545cc2dd95bc3d27f056016300673bdeee78f5"
install_pinned_node "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git" \
  "$COMFY/custom_nodes/ComfyUI_UltimateSDUpscale" \
  "a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df"
install_pinned_node "https://github.com/wallish77/wlsh_nodes.git" \
  "$COMFY/custom_nodes/wlsh_nodes" \
  "97807467bf7ff4ea01d529fcd6e666758f34e3c1"
install_pinned_node "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" \
  "$COMFY/custom_nodes/ComfyUI-VideoHelperSuite" \
  "4ee72c065db22c9d96c2427954dc69e7b908444b"
install_pinned_node "https://github.com/kijai/ComfyUI-KJNodes.git" \
  "$COMFY/custom_nodes/ComfyUI-KJNodes" \
  "e27a505b3ba6ce42687fe00500deda103d9d6071"
install_pinned_node "https://github.com/ClownsharkBatwing/RES4LYF.git" \
  "$COMFY/custom_nodes/RES4LYF" \
  "419de2d7c78f415dde9aa352a7231820ebfc17a4"

# This tiny local node makes one checkbox load the selected high- and low-noise
# pair internally. The inactive format may therefore be absent or downloading.
SWITCH_NODE_DIR="$COMFY/custom_nodes/ComfyUI-Wan22-Model-Pair-Switch"
SWITCH_NODE_FILE="$SWITCH_NODE_DIR/__init__.py"
SWITCH_NODE_PART="$SWITCH_NODE_DIR/__init__.part.py"
if [ -n "${WAN22_SWITCH_NODE_URL:-}" ]; then
  SWITCH_NODE_URL="$WAN22_SWITCH_NODE_URL"
elif [ -n "$SCRIPT_BASE_URL" ]; then
  SWITCH_NODE_URL="$SCRIPT_BASE_URL/wan22_model_pair_switch.py"
else
  echo "FATAL: Set WAN22_SWITCH_NODE_URL when running this setup script directly."
  exit 1
fi

echo "Installing/updating WAN 2.2 staged model-pair switch..."
mkdir -p "$SWITCH_NODE_DIR"
curl -fsSL --retry 5 --retry-delay 2 "$SWITCH_NODE_URL" -o "$SWITCH_NODE_PART"
python -m py_compile "$SWITCH_NODE_PART"
mv "$SWITCH_NODE_PART" "$SWITCH_NODE_FILE"
echo "Installed WAN 2.2 staged model-pair switch."

# Assets for the frame-to-frame branch only. Both Q8 GGUF and FP8 safetensor
# I2V pairs are downloaded so either model format is available.
DOWNLOAD_KINDS=(
  "MODEL" "MODEL" "MODEL" "MODEL" "TE" "VAE" "LORA" "LORA"
)
DOWNLOAD_URLS=(
  "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/c95ab6c210a60ff915aa3f7cb0fa07300b0b2f36/wan2.2_i2v_high_noise_14B_Q8_0.gguf"
  "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/c95ab6c210a60ff915aa3f7cb0fa07300b0b2f36/wan2.2_i2v_low_noise_14B_Q8_0.gguf"
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/fb1388adc906ab39ffc26ee40e96b22886b56bc4/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/fb1388adc906ab39ffc26ee40e96b22886b56bc4/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/06e001fc51048fb03433a6fb25334de7836704a5/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/06e001fc51048fb03433a6fb25334de7836704a5/split_files/vae/wan_2.1_vae.safetensors"
  "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/570044187a5219776ef30a5c60c6f76428a3a10a/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
  "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/570044187a5219776ef30a5c60c6f76428a3a10a/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)
DOWNLOAD_OUTPUTS=(
  "$COMFY/models/diffusion_models/Wan2.2/wan2.2_i2v_high_noise_14B_Q8_0.gguf"
  "$COMFY/models/diffusion_models/Wan2.2/wan2.2_i2v_low_noise_14B_Q8_0.gguf"
  "$COMFY/models/diffusion_models/Wan2.2/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
  "$COMFY/models/diffusion_models/Wan2.2/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
  "$COMFY/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "$COMFY/models/vae/wan_2.1_vae.safetensors"
  "$COMFY/models/loras/Wan2.2/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
  "$COMFY/models/loras/Wan2.2/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)
DOWNLOAD_SIZES=()

for i in "${!DOWNLOAD_URLS[@]}"; do
  size=""
  if [ ! -s "${DOWNLOAD_OUTPUTS[$i]}" ]; then
    size="$(remote_size "${DOWNLOAD_URLS[$i]}")"
  fi
  DOWNLOAD_SIZES[$i]="$size"
done

# Q8 is the default workflow format. Download it and all shared assets before
# ComfyUI starts. The optional FP8 pair is launched in the background below.
FOREGROUND_DOWNLOADS=(0 1 4 5 6 7)
FP8_DOWNLOADS=(2 3)
download_group "WAN22-READY" "${FOREGROUND_DOWNLOADS[@]}"

WORKFLOW_DIR="$COMFY/user/default/workflows"
WORKFLOW_NAME="WAN2.2_base_Q8_max_realism_20H20L.json"
WORKFLOW_FILE="$WORKFLOW_DIR/$WORKFLOW_NAME"
if [ -n "${WAN22_WORKFLOW_URL:-}" ]; then
  WORKFLOW_URL="$WAN22_WORKFLOW_URL"
elif [ -n "${SETUP_SCRIPT_URL:-}" ]; then
  WORKFLOW_URL="$SCRIPT_BASE_URL/$WORKFLOW_NAME"
else
  echo "FATAL: Set WAN22_WORKFLOW_URL when running this setup script directly."
  exit 1
fi

if [ -s "$WORKFLOW_FILE" ]; then
  echo "Ready WORKFLOW  $WORKFLOW_NAME (already installed)"
else
  echo "Installing WORKFLOW  $WORKFLOW_NAME"
  mkdir -p "$WORKFLOW_DIR"
  curl -fsSL --retry 5 --retry-delay 2 "$WORKFLOW_URL" -o "${WORKFLOW_FILE}.part"
  python3 -m json.tool "${WORKFLOW_FILE}.part" >/dev/null
  mv "${WORKFLOW_FILE}.part" "$WORKFLOW_FILE"
  echo "Installed  WORKFLOW  $WORKFLOW_NAME"
fi

echo "=== WAN 2.2 foreground assets and workflow are ready ==="
echo "Patching original /start.sh for custom FileBrowser credentials..."
# This retains runtime expansion of FILEBROWSER_* and leaves JUPYTER_PASSWORD
# to the original /start.sh; neither credential is written into this script.

python3 - <<'PY_PATCH_START'
from pathlib import Path

p = Path('/start.sh')
s = p.read_text()

old = 'filebrowser users add admin adminadmin12 --perm.admin'
new = 'filebrowser users add "${FILEBROWSER_USERNAME:-admin}" "${FILEBROWSER_PASSWORD:-adminadmin12}" --perm.admin'

if old in s:
    s = s.replace(old, new)
    print('Patched FileBrowser credentials line in /start.sh.')
elif new in s:
    print('/start.sh already has custom FileBrowser credential support.')
else:
    print('WARNING: Could not find FileBrowser credentials line in /start.sh; leaving it unchanged.')

p.write_text(s)
PY_PATCH_START

if [ -n "${NTFY_TOPIC:-}" ]; then
  echo "Watching for ComfyUI to become ready before sending ntfy notification..."
  (
    trap '' HUP
    for _ in $(seq 1 180); do
      if curl -fsS --max-time 2 http://127.0.0.1:8188/ >/dev/null 2>&1; then
        notify_ntfy \
          "ComfyUI is ready" "high" "tada" \
          "ComfyUI started successfully and is accepting connections on port 8188."
        exit 0
      fi
      sleep 5
    done
    echo "WARNING: ComfyUI did not become ready within 15 minutes; no ready notification sent."
  ) &
fi

start_fp8_downloads

echo "Returning to wrapper. SageAttention bootstrap will run next."
echo "ComfyUI will start with Q8 ready while the optional FP8 pair continues downloading."
exit 0
