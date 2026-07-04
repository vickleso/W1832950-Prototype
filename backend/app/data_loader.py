import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor


class XFactaDataset(Dataset):
    # This dataset class loads processed JSONL examples into the format expected by the Qwen fine-tuning pipeline.
    
    def __init__(self, data_dir, max_samples=500):
        # Load the processor that prepares data for the model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
        self.data = []
        self.max_samples = max_samples
        
        print(f"Loading data from {data_dir}...\n")
        
        # Find all JSONL files in the directory
        all_files = list(Path(data_dir).rglob("*.jsonl"))
        print(f"Found {len(all_files)} files\n")
        
        # Read each file
        for file_path in all_files:
            print(f"Reading {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # Only keep examples that have text and label
                        if 'text' in example and 'label' in example:
                            self.data.append(example)
        
        print(f"\nTotal examples loaded: {len(self.data)}\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get one example
        example = self.data[idx]
        
        text = example['text']
        images = example.get('images', [])
        label = example['label']
        
        # Create the answer
        answer = "Real" if label == True else "Misinformation"
        
        # Create a conversation (how Qwen3-VL expects input)
        conversation = [
            {"role": "user", "content": f"You are a misinformation detection assistant, you are to detect if the following text is real or misinformation:\n{text}"},
            {"role": "assistant", "content": answer}
        ]
        
        # Format the conversation
        formatted_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Load image if it exists
        image = None
        if 'images' in example and example['images']:
            img_path = example['images'][0]
            if Path(img_path).exists():
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        # Process text and image together
        if image:
            inputs = self.processor(
                text=[formatted_text],
                images=[image],
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[formatted_text],
                return_tensors="pt"
            )
        
        # Remove extra batch dimension
        inputs = {k: v[0] for k, v in inputs.items()}
        
        # The labels are the same as input_ids (model learns to predict next token)
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs


def pad_batch(batch):
    # This helper pads batches so the so they are all equal lengths for the model to train on.
    
    
    result = {}

    def _pad_tensor(tensor: torch.Tensor, pad_width: int, pad_value: int) -> torch.Tensor:
        if pad_width <= 0:
            return tensor

        pad_shape = tensor.shape[:-1] + (pad_width,)
        pad_block = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad_block], dim=-1)

    for key in batch[0].keys():
        tensors = [item[key] for item in batch]

        # Find the longest sequence (works for 1D and tensors with variable last-dim)
        max_length = max(t.shape[-1] for t in tensors)

        padded = []
        for tensor in tensors:
            padding_needed = max_length - tensor.shape[-1]

            pad_value = -100 if key == 'labels' else 0
            padded_tensor = _pad_tensor(tensor, padding_needed, pad_value)
            padded.append(padded_tensor)

        result[key] = torch.stack(padded)

    return result


def get_loader(data_dir, batch_size, max_samples=500):
    # A training loader that feeds the data into the fine-tuning loop.
    dataset = XFactaDataset(data_dir, max_samples)
    
    if len(dataset) == 0:
        raise ValueError("No data found!")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_batch,
        drop_last=True,
        num_workers=0,
    )

"""
Was adapted from example found at :
- https://unsloth.ai/docs (Unsloth Docs | Unsloth Documentation, 2026)
- https://huggingface.co/docs (Hugging Face - Documentation, no date)
"""