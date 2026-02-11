import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path


from data_loader import get_loader

# Settings
MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
DATA_DIR = "../../data/processed"
SAVE_DIR = Path("../../models/qwen_finetuned")
DEVICE = "cuda"
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
MAX_SAMPLES = 500

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def add_lora(model):
    """Make model trainable with fewer parameters"""
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        ]
    )
    return get_peft_model(model, config)


def train():
    print("\nTRAINING STARTED\n")
    
    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = add_lora(model)
    print("Model ready\n")
    
    # Load data
    print("Loading data...")
    loader = get_loader(DATA_DIR, BATCH_SIZE, MAX_SAMPLES)
    print(f"{len(loader)} batches per epoch\n")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Train
    print("Training...\n")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        
        for batch_num, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            # Move to GPU
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Train step
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item()
            
            # Free memory every 50 batches
            if (batch_num + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f}\n")
        
        # Save
        save_path = SAVE_DIR / f"epoch-{epoch+1}"
        save_path.mkdir(exist_ok=True)
        model.save_pretrained(str(save_path))
        processor.save_pretrained(str(save_path))
        print(f"Saved to {save_path}\n")
    
    # Save final
    final = SAVE_DIR / "final"
    final.mkdir(exist_ok=True)
    model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    
    print("DONE!")
    print(f"Model saved to {final}\n")


if __name__ == "__main__":
    train()
