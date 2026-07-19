class Wan22ModelPairSwitch:
    """Lazily select the GGUF or FP8 WAN 2.2 high/low model pair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_high": ("MODEL", {"lazy": True}),
                "gguf_low": ("MODEL", {"lazy": True}),
                "fp8_high": ("MODEL", {"lazy": True}),
                "fp8_low": ("MODEL", {"lazy": True}),
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
    FUNCTION = "select_pair"
    CATEGORY = "WAN 2.2"

    def check_lazy_status(
        self,
        gguf_high=None,
        gguf_low=None,
        fp8_high=None,
        fp8_low=None,
        use_fp8=False,
    ):
        if use_fp8:
            return [
                name
                for name, value in (("fp8_high", fp8_high), ("fp8_low", fp8_low))
                if value is None
            ]
        return [
            name
            for name, value in (("gguf_high", gguf_high), ("gguf_low", gguf_low))
            if value is None
        ]

    def select_pair(
        self,
        gguf_high=None,
        gguf_low=None,
        fp8_high=None,
        fp8_low=None,
        use_fp8=False,
    ):
        if use_fp8:
            return (fp8_high, fp8_low)
        return (gguf_high, gguf_low)


NODE_CLASS_MAPPINGS = {
    "Wan22ModelPairSwitch": Wan22ModelPairSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22ModelPairSwitch": "WAN 2.2 Model Format (Q8 GGUF / FP8)",
}
