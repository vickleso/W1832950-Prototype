import os
import json
import re
import traceback
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
import requests
from dotenv import load_dotenv
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

# These helpers designate the local model paths so the backend can load trained weights from the repository.

def _resolve_repo_path(env_name, default_path):
    env_value = os.getenv(env_name)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            return (REPO_ROOT / candidate).resolve()
        return candidate
    return default_path


LOCAL_MODEL_DIR = _resolve_repo_path(
    "MODEL_DIR", REPO_ROOT / "models" / "qwen_finetuned" / "final"
)
TWHIN_MODEL_DIR = _resolve_repo_path(
    "TWHIN_MODEL_DIR", REPO_ROOT / "TwHIN-BERT-Misinformation-Classifier"
)


def _safe_load_image(image_url):
    # This utility loads the images of the post from URLs so the vision model can analyse the attached media when available.
    if not image_url:
        return Image.new("RGB", (448, 448), color=(200, 200, 200))

    try:
        if str(image_url).startswith("http"):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        if Path(image_url).exists():
            return Image.open(image_url).convert("RGB")

    except Exception as error:
        print(f"Warning: Failed to load image {image_url}: {error}")

    return Image.new("RGB", (448, 448), color=(200, 200, 200))


def _build_verdict(classification, confidence=0.0, misinformation_prob=None):
    # This helper translates the raw model scores into a verdict and status for the API response.
    if misinformation_prob is not None:
        fake_likelihood = float(misinformation_prob)
    elif classification == "Real":
        fake_likelihood = 1.0 - float(confidence)
    else:
        fake_likelihood = float(confidence)

    fake_likelihood = max(0.0, min(1.0, fake_likelihood))

    if fake_likelihood >= 0.75:
        verdict = "Likely misinformation"
        status = "misinformation"
    elif fake_likelihood <= 0.4:
        verdict = "Likely true"
        status = "true"
    else:
        verdict = "Unsure"
        status = "unsure"

    return verdict, status, fake_likelihood


def _build_reasoning(classification, confidence, model_name, factual_prob=None, misinformation_prob=None):
    # This helper creates a short explanation for the classification given.
    score = int(confidence * 100)
    if classification == "Misinformation":
        return (
            f"The {model_name} model found a greater likelihood of misleading content in the post, "
            f"so it classified the message as misinformation with {score}% confidence."
        )

    if classification == "Real":
        return (
            f"The {model_name} model found the content more consistent with factual language, "
            f"so it classified the post as real with {score}% confidence."
        )

    if misinformation_prob is not None and factual_prob is not None:
        return (
            f"The {model_name} model is uncertain because the probabilities are close: "
            f"{misinformation_prob:.0%} misinformation and {factual_prob:.0%} factual."
        )

    return f"The {model_name} model returned an uncertain result with {score}% confidence."


def _parse_qwen_output(output_text):
    # This helper extracts labels, confidence, and explanations from the Qwen model's text output.
    label = None
    confidence = None
    explanation = None
    text = output_text.strip()

    try:
        json_match = re.search(r"\{.*?\}", text, re.S)
        if json_match:
            text = json_match.group(0)
        data = json.loads(text)
        label = data.get("label")
        confidence = float(data.get("confidence", 0)) if data.get("confidence") is not None else None
        explanation = data.get("explanation")
    except Exception:
        if re.search(r"\bmisinformation\b", text, re.I):
            label = "Misinformation"
        elif re.search(r"\breal\b|\btrue\b|\bfactual\b", text, re.I):
            label = "Real"

        percent_match = re.search(r"([0-9]{1,3})\s*%", text)
        if percent_match:
            percent = float(percent_match.group(1))
            confidence = max(0.0, min(1.0, percent / 100.0))
        else:
            float_match = re.search(r"([01](?:\.\d+)|0?\.\d+)", text)
            if float_match:
                candidate = float(float_match.group(1))
                if 0.0 <= candidate <= 1.0:
                    confidence = candidate

        if explanation is None:
            expl_match = re.search(r"explanation\s*[:\-]\s*(.+)$", text, re.I | re.S)
            if expl_match:
                explanation = expl_match.group(1).strip()

    if label is None:
        label = "Misinformation" if "misinformation" in text.lower() else "Real"

    if confidence is None:
        confidence = 0.75 if label in {"Misinformation", "Real"} else 0.5

    return label, confidence, explanation


class QwenVLDetector:
    # This class allows the Qwen vision model to classify posts and generate a short explanation for the misinformation results.

    def __init__(self):
        print("Loading Qwen3-VL model...")
        try:
            base_model_name = "Qwen/Qwen3-VL-4B-Thinking"
    
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

            if LOCAL_MODEL_DIR.exists():
                self.model = PeftModel.from_pretrained(self.model, LOCAL_MODEL_DIR, device_map="cpu")
            else:
                raise FileNotFoundError(f"Missing fine-tuned model at {LOCAL_MODEL_DIR}")

            self.processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
            self.model.eval()
            print("Qwen3-VL model loaded successfully.\n")
        except Exception as error:
            print(f"Failed to load Qwen3-VL: {error}")
            traceback.print_exc()
            raise

    def analyse(self, text, image_url=None):
        if not text or not isinstance(text, str):
            return {
                "classification": "Error",
                "confidence": 0.0,
                "reasoning": "Invalid input text.",
                "raw": "",
            }

        image = _safe_load_image(image_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are a misinformation detector. Respond with JSON only, using exactly the keys "
                            "\"label\" (Misinformation or Real), \"confidence\" (0.00-1.00), and \"explanation\".\n"
                            "Use one short sentence for explanation and do not include reasoning steps, "
                            "assistant tags, or extra text.\n"
                            f"Text: {text}"
                        ),
                    },
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"<image>\nUser: Is this misinformation? Please explain briefly why you made the classification.\n"
                f"{text}\nAssistant:"
            )

        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate( # type: ignore
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=400,
                do_sample=False,
                num_beams=1,
            )

        input_length = inputs.get("input_ids").shape[1]
        generated_ids = outputs[0][input_length:]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        label, confidence, explanation = _parse_qwen_output(raw_text)

        reasoning = explanation or _build_reasoning(label, confidence, "Qwen3-VL")
        verdict, status, fake_likelihood = _build_verdict(label, confidence)

        return {
            "classification": label,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "raw": raw_text,
            "verdict": verdict,
            "status": status,
            "fake_likelihood": round(fake_likelihood, 4),
            "details": {
                "model_output": raw_text,
            },
        }


class TwHINDetector:
    # This class wraps the text-only TwHIN model for fast misinformation classification so that it can analyse the post without images.

    def __init__(self):
        print("Loading TwHIN-BERT model...")
        try:
            if not TWHIN_MODEL_DIR.exists():
                raise FileNotFoundError(f"Missing TwHIN model at {TWHIN_MODEL_DIR}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(TWHIN_MODEL_DIR))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(TWHIN_MODEL_DIR)
            )
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            self.label_map = self.model.config.id2label
            print(f"TwHIN labels: {self.label_map}")
            print(f"TwHIN-BERT model loaded on {self.device}.\n")
        except Exception as error:
            print(f"Failed to load TwHIN-BERT: {error}")
            traceback.print_exc()
            raise

    def analyse(self, text):
        if not text or not isinstance(text, str):
            return {
                "classification": "Error",
                "confidence": 0.0,
                "reasoning": "Invalid input text.",
                "raw": "",
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        factual_prob = probabilities[0][0].item()
        misinformation_prob = probabilities[0][1].item()

        classification = (
            "Misinformation" if misinformation_prob >= 0.5 else "Real"
        )
        confidence = misinformation_prob if classification == "Misinformation" else factual_prob

        reasoning = _build_reasoning(
            classification,
            confidence,
            "TwHIN-BERT",
            factual_prob=factual_prob,
            misinformation_prob=misinformation_prob,
        )
        verdict, status, fake_likelihood = _build_verdict(
            classification, misinformation_prob=misinformation_prob
        )

        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "raw": (
                f"Classification: {classification}. "
                f"Misinformation probability {misinformation_prob:.2%}."
            ),
            "details": {
                "factual_prob": round(factual_prob, 4),
                "misinformation_prob": round(misinformation_prob, 4),
            },
            "verdict": verdict,
            "status": status,
            "fake_likelihood": round(fake_likelihood, 4),
        }

class Detector(QwenVLDetector):
    """Legacy alias for QwenVLDetector."""
    pass

"""
In this file I used Github Copilot to help write the code for the QwenVLDetector and TwHINDetector classes. 
The code was generated based on the requirements of loading the models, processing inputs, and generating outputs. 
I reviewed and modified the generated code to ensure it met our specific needs for misinformation detection, paraphrase of prompts entered:

- "My model keeps using the confidence score to classify the posts, can you help me find the issue and provide potential solutions?"

- "What is a good way to structure my how my two models work that generalises a big portion of the code and makes it easier to maintain?"

- "How do I make my model explanation more detailed and dynamic, reflecting the reasoning behind the classification?"

The code was also adapted from examples found at: 
- https://unsloth.ai/docs (Unsloth Docs | Unsloth Documentation, 2026)
- https://huggingface.co/docs (Hugging Face - Documentation, no date)

"""