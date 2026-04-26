import os
import torch
import folder_paths
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _scan_qwen_dirs():
    """text_encoders 하위 디렉터리 중 HuggingFace 모델(config.json 포함)을 반환."""
    dirs = []
    for base in folder_paths.get_folder_paths("text_encoders"):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            candidate = os.path.join(base, name)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                dirs.append(name)
    return dirs if dirs else ["(no models found)"]


class QwenModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": (_scan_qwen_dirs(),),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "translator"
    TITLE = "Load Qwen Model"

    # 같은 모델 경로면 재로드하지 않도록 캐시
    _cache: dict = {}

    def load_model(self, model_dir, dtype, quantization, device):
        cache_key = (model_dir, dtype, quantization, device)
        if cache_key in QwenModelLoader._cache:
            print(f"[QwenLoader] Cache hit: {model_dir}")
            return (QwenModelLoader._cache[cache_key],)

        model_path = None
        for base in folder_paths.get_folder_paths("text_encoders"):
            candidate = os.path.join(base, model_dir)
            if os.path.isdir(candidate):
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(
                f"[QwenLoader] '{model_dir}' not found in text_encoders directories."
            )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_map[dtype],
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[QwenLoader] Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"[QwenLoader] Loading model ({dtype}, quant={quantization}, device={device}) from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[dtype],
            quantization_config=bnb_config,
            device_map=device,
        )
        model.eval()
        print(f"[QwenLoader] Model ready: {model_dir}")

        result = {
            "model": model,
            "tokenizer": tokenizer,
            "model_path": model_path,
            "model_dir": model_dir,
        }
        QwenModelLoader._cache[cache_key] = result
        return (result,)


class QwenTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "system_prompt": ("STRING", {"multiline": True, "default": "Translate the following Korean text into an English prompt suitable for AI image generation. Output only the English prompt."}),
                "korean_text": ("STRING", {"multiline": True}),
                "max_new_tokens": ("INT", {"default": 200, "min": 32, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("english_prompt",)
    FUNCTION = "translate"
    CATEGORY = "translator"
    TITLE = "Qwen Translator (KO→EN)"

    @torch.inference_mode()
    def translate(self, qwen_model, system_prompt, korean_text, max_new_tokens):
        model = qwen_model["model"]
        tokenizer = qwen_model["tokenizer"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": korean_text},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return (result,)


class ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "translator"
    TITLE = "Show Text"

    def show(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "QwenModelLoader": QwenModelLoader,
    "QwenTranslator": QwenTranslator,
    "ShowText": ShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenModelLoader": "Load Qwen Model",
    "QwenTranslator": "Qwen Translator (KO→EN)",
    "ShowText": "Show Text",
}
