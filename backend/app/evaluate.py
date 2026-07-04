import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from inference import QwenVLDetector, TwHINDetector

# =========================
# 3. Paths
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_JSON_PATH = ROOT_DIR / "test.json"
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", ""))
OUTPUT_DIR = ROOT_DIR
OUTPUT_CSV = OUTPUT_DIR / "qwen3vl_eval_predictions.csv"


def normalize_prediction(text):
    if text is None:
        return None

    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z ]+", " ", text)
    text = " ".join(text.split())

    if "real" in text and "fake" not in text:
        return 1
    if "fake" in text and "real" not in text:
        return 0
    if text.startswith("real"):
        return 1
    if text.startswith("fake"):
        return 0

    return None


def label_to_int(label):
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, str):
        x = label.strip().lower()
        if x in {"true", "real"}:
            return 1
        if x in {"false", "fake"}:
            return 0
    return int(label)


def load_first_existing_image(images, image_root):
    if not images:
        return None

    if isinstance(images, str):
        images = [images]

    for img_path in images:
        candidate = Path(img_path)
        if candidate.exists():
            return candidate
        if image_root:
            candidate = image_root / img_path
            if candidate.exists():
                return candidate

    return None


def predict_one(detector, image_path, post_text):
    if isinstance(detector, TwHINDetector):
        return detector.analyse(post_text)
    return detector.analyse(post_text, image_path)


def build_dataset(json_path, image_root):
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    records = []
    skipped = 0

    for item in examples:
        text = str(item.get("text", "")).strip()
        images = item.get("images", [])
        label = item.get("label", None)

        if label is None or not text:
            skipped += 1
            continue

        image_path = load_first_existing_image(images, image_root)
        if image_path is None:
            skipped += 1
            continue

        records.append(
            {
                "text": text,
                "image_path": str(image_path),
                "true_label": label_to_int(label),
                "raw_images": images,
            }
        )

    df = pd.DataFrame(records)
    return df, skipped


def evaluate_dataframe(detector, df, limit=None):
    y_true = []
    y_pred = []
    raw_outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if limit is not None and len(y_true) >= limit:
            break

        try:
            result = predict_one(detector, row["image_path"], row["text"])
            raw_text = result.get("raw", "")
            classification = result.get("classification", "")

            pred_label = normalize_prediction(classification)
            if pred_label is None:
                pred_label = normalize_prediction(raw_text)

            raw_outputs.append(raw_text)
            if pred_label is None:
                continue

            y_true.append(int(row["true_label"]))
            y_pred.append(int(pred_label))
        except Exception as exc:
            raw_outputs.append(f"ERROR: {exc}")
            continue

    return y_true, y_pred, raw_outputs


def print_metrics(y_true, y_pred):
    print("Evaluation results")
    print("==================")
    print(f"Evaluated samples: {len(y_true)}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("Confusion Matrix")
    print("================")
    print(confusion_matrix(y_true, y_pred))

    print("Classification Report")
    print("=====================")
    print(classification_report(y_true, y_pred, target_names=["fake", "real"], zero_division=0))


def save_predictions(df, raw_outputs, path):
    out_rows = []
    for idx, row in enumerate(df.itertuples(index=False)):
        if idx >= len(raw_outputs):
            break
        out_rows.append(
            {
                "text": row.text,
                "image_path": row.image_path,
                "true_label": row.true_label,
                "raw_prediction": raw_outputs[idx],
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TwHIN and Qwen models on the dataset.")
    parser.add_argument(
        "--test-file",
        default=TEST_JSON_PATH,
        help="Path to the test JSON file.",
    )
    parser.add_argument(
        "--model",
        choices=["twhin", "qwen", "both"],
        default="both",
        help="Which model to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to evaluate.")
    parser.add_argument(
        "--image-root",
        default=os.getenv("IMAGE_ROOT", ""),
        help="Optional root folder to resolve image paths.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.test_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test file does not exist: {dataset_path}")

    image_root = Path(args.image_root) if args.image_root else IMAGE_ROOT
    df, skipped = build_dataset(dataset_path, image_root)
    print(f"Loaded {len(df)} valid samples from {dataset_path}")
    print(f"Skipped {skipped} invalid or missing-image samples")

    if args.model in {"twhin", "both"}:
        print("Evaluating TwHIN-BERT...")
        tw_model = TwHINDetector()
        y_true, y_pred, raw_outputs = evaluate_dataframe(tw_model, df, limit=args.limit)
        print_metrics(y_true, y_pred)
        save_predictions(df, raw_outputs, OUTPUT_DIR / "twhin_eval_predictions.csv")
        print(f"Saved predictions to {OUTPUT_DIR / 'twhin_eval_predictions.csv'}")

    if args.model in {"qwen", "both"}:
        print("Evaluating Qwen3-VL...")
        qwen_model = QwenVLDetector()
        y_true, y_pred, raw_outputs = evaluate_dataframe(qwen_model, df, limit=args.limit)
        print_metrics(y_true, y_pred)
        save_predictions(df, raw_outputs, OUTPUT_CSV)
        print(f"Saved predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


"""
This script was written with the assistance of Github Copilot. It is designed to evaluate the performance of TwHIN and Qwen3-VL models on a given dataset. 
The script loads a test dataset from a JSON file, processes the data, and evaluates the models' predictions against the true labels. 
It computes various metrics such as accuracy, precision, recall, F1 score, confusion matrix, and classification report. \
The predictions are saved to CSV files for further analysis.
"""