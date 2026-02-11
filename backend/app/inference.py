import os
from pathlib import Path
from io import BytesIO
import traceback
import torch
from PIL import Image
import requests
from dotenv import load_dotenv
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

# Environment / paths
ENV_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-4B-Thinking")
LOCAL_MODEL_DIR = Path(os.getenv("MODEL_DIR", "../../models/qwen_finetuned/final"))

class Detector:
    def __init__(self, use_finetuned=True):
        print("Loading model...")
        try:
            # Load base model first
            base_model_name = "Qwen/Qwen3-VL-4B-Thinking"
            print(f"Loading base model: {base_model_name}")
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

            # Load base model
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load adapter if finetuned dir exists
            if use_finetuned and LOCAL_MODEL_DIR.exists():
                print(f"Loading PEFT adapter from {LOCAL_MODEL_DIR}")
                self.model = PeftModel.from_pretrained(self.model, LOCAL_MODEL_DIR)
                print("Adapter loaded successfully")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
            
            self.model.eval()
            print("Model ready\n")
            
        except Exception as e:
            print("Warning: Could not load models:", e)
            traceback.print_exc()
            raise

    def analyse(self, text, image_url=None):
        # Create a placeholder neutral image
        image = Image.new('RGB', (448, 448), color=(200, 200, 200))

        # Try to load image if URL provided
        if image_url:
            try:
                if str(image_url).startswith("http"):
                    resp = requests.get(image_url, timeout=10)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content)).convert("RGB")
                elif Path(image_url).exists():
                    image = Image.open(image_url).convert("RGB")
            except Exception as e:
                print(f"Could not load image ({image_url}): {e}")

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Is this misinformation?\n{text}"}
        ]}]
        
        try:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback: simple prompt with image token
            prompt = f"<image>\nUser: Is this misinformation?\n{text}\nAssistant:"

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )

        # Move tensors to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.get('input_ids'),
                attention_mask=inputs.get('attention_mask'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                max_new_tokens=40,
                do_sample=False
            )

        # Decode
        try:
            result = self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Decode error: {e}")
            result = str(outputs[0])

        print("DEBUG raw model output:", result)

        # try to parse JSON response like: {"label":"Misinformation","confidence":0.78,"explanation":"..."}
        import json, re
        label = None
        confidence = None
        explanation = None

        # first try direct JSON
        try:
            j = json.loads(result.strip())
            label = j.get("label")
            confidence = float(j.get("confidence", 0)) if j.get("confidence") is not None else None
            explanation = j.get("explanation")
        except Exception:
            # fallback: find label words and a numeric confidence
            if re.search(r"\bmisinformation\b", result, re.I):
                label = "Misinformation"
            elif re.search(r"\breal\b|\btrue\b|\bfactual\b", result, re.I):
                label = "Real"
            m = re.search(r"([01](?:\.\d+)|0?\.\d+)", result)
            if m:
                confidence = float(m.group(1))

        # final fallback
        if label is None:
            label = "Misinformation" if 'misinformation' in (result or "").lower() else "Real"
        if confidence is None:
            confidence = 0

        return {
            'classification': label,
            'confidence': confidence,
            'raw': result,
            'explanation': explanation
        }

