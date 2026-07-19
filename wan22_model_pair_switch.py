import os

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


NODE_CLASS_MAPPINGS = {
    "Wan22ModelPairSwitch": Wan22ModelPairSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22ModelPairSwitch": "WAN 2.2 Model Format (Q8 GGUF / FP8)",
}
