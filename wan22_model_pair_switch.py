import os

import comfy.sd
import comfy.utils
import folder_paths
import nodes


class Wan22ModelPairSwitch:
    """Load only the selected WAN 2.2 high/low model pair.

    Keeping the loaders inside this node means ComfyUI can start and validate
    GGUF prompts while the optional FP8 files are still downloading.
    """

    GGUF_HIGH = "Wan2.2/wan2.2_i2v_high_noise_14B_Q8_0.gguf"
    GGUF_LOW = "Wan2.2/wan2.2_i2v_low_noise_14B_Q8_0.gguf"
    FP8_HIGH = "Wan2.2/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    FP8_LOW = "Wan2.2/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_fp8": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "FP8 SAFETENSORS",
                        "label_off": "Q8 GGUF",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("high_noise_model", "low_noise_model")
    FUNCTION = "load_pair"
    CATEGORY = "WAN 2.2"

    @staticmethod
    def _require_file(folder_name, relative_name, format_name):
        path = folder_paths.get_full_path(folder_name, relative_name)
        if path is None or not os.path.isfile(path):
            if format_name == "FP8":
                raise FileNotFoundError(
                    "The WAN 2.2 FP8 pair is still downloading or failed to "
                    "download. Leave MODEL FORMAT set to Q8 GGUF for now. "
                    "Background status: "
                    "/workspace/runpod-slim/wan22-fp8-download.status"
                )
            raise FileNotFoundError(
                f"Required WAN 2.2 {format_name} model is missing: {relative_name}"
            )

    @staticmethod
    def _loader(class_name):
        loader_class = nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if loader_class is None:
            raise RuntimeError(
                f"Required loader node {class_name} is not installed or failed to import."
            )
        return loader_class()

    def load_pair(self, use_fp8=False):
        if use_fp8:
            self._require_file("diffusion_models", self.FP8_HIGH, "FP8")
            self._require_file("diffusion_models", self.FP8_LOW, "FP8")
            loader = self._loader("UNETLoader")
            high = loader.load_unet(self.FP8_HIGH, "default")[0]
            low = loader.load_unet(self.FP8_LOW, "default")[0]
            return (high, low)

        self._require_file("unet_gguf", self.GGUF_HIGH, "Q8 GGUF")
        self._require_file("unet_gguf", self.GGUF_LOW, "Q8 GGUF")
        loader = self._loader("UnetLoaderGGUF")
        high = loader.load_unet(self.GGUF_HIGH)[0]
        low = loader.load_unet(self.GGUF_LOW)[0]
        return (high, low)


class Wan22LightXLoRAPair:
    """Apply the WAN 2.2 I2V LightX pair without dropping modulation deltas.

    The 1022 LightX files contain ``*.diff_m`` tensors. LightX2V's official
    ComfyUI-WanVideoWrapper workflow renames those keys to
    ``*.modulation.diff`` before handing the state dict to ComfyUI's LoRA
    patcher. Generic LoRA nodes do not perform that WAN-specific conversion,
    which leaves one modulation delta unloaded for every transformer block.
    """

    HIGH_LORA = (
        "Wan2.2/"
        "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    )
    LOW_LORA = (
        "Wan2.2/"
        "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": ("MODEL",),
                "low_noise_model": ("MODEL",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "LIGHTX ON",
                        "label_off": "LIGHTX OFF",
                    },
                ),
                "high_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "low_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("high_noise_model", "low_noise_model")
    FUNCTION = "apply_pair"
    CATEGORY = "WAN 2.2"

    def __init__(self):
        self._lora_cache = {}

    @staticmethod
    def _lora_path(relative_name):
        path = folder_paths.get_full_path("loras", relative_name)
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required WAN 2.2 LightX LoRA is missing: {relative_name}"
            )
        return path

    @staticmethod
    def _normalize_lightx_keys(lora_state):
        normalized = {}
        renamed = 0
        for key, value in lora_state.items():
            normalized_key = key.replace(".diff_m", ".modulation.diff")
            if normalized_key != key:
                renamed += 1
            if normalized_key in normalized:
                raise ValueError(
                    "LightX key normalization produced a duplicate tensor: "
                    f"{normalized_key}"
                )
            normalized[normalized_key] = value

        if renamed == 0:
            raise ValueError(
                "The selected file contains no LightX .diff_m tensors; refusing "
                "to silently apply an unexpected LoRA through this specialized node."
            )
        return normalized, renamed

    def _load_lora(self, relative_name):
        path = self._lora_path(relative_name)
        modified_ns = os.stat(path).st_mtime_ns
        cached = self._lora_cache.get(path)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]

        raw_state = comfy.utils.load_torch_file(path, safe_load=True)
        normalized, renamed = self._normalize_lightx_keys(raw_state)
        self._lora_cache[path] = (modified_ns, normalized)
        print(
            f"[WAN 2.2 LightX] normalized {renamed} .diff_m modulation "
            f"tensors in {os.path.basename(path)}"
        )
        return normalized

    def _apply_lora(self, model, relative_name, strength):
        lora_state = self._load_lora(relative_name)
        patched_model, _ = comfy.sd.load_lora_for_models(
            model, None, lora_state, strength, 0.0
        )
        return patched_model

    def apply_pair(
        self,
        high_noise_model,
        low_noise_model,
        enabled=False,
        high_strength=1.0,
        low_strength=1.0,
    ):
        if not enabled:
            return (high_noise_model, low_noise_model)

        high = self._apply_lora(high_noise_model, self.HIGH_LORA, high_strength)
        low = self._apply_lora(low_noise_model, self.LOW_LORA, low_strength)
        return (high, low)


NODE_CLASS_MAPPINGS = {
    "Wan22ModelPairSwitch": Wan22ModelPairSwitch,
    "Wan22LightXLoRAPair": Wan22LightXLoRAPair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22ModelPairSwitch": "WAN 2.2 Model Format (Q8 GGUF / FP8)",
    "Wan22LightXLoRAPair": "WAN 2.2 LightX LoRA Pair (Full Keys)",
}
