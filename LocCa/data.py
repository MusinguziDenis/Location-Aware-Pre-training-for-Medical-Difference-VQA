import os
from torch.utils.data import Dataset, DataLoader
import random
import torch
from torchvision.transforms import v2
from PIL import Image
from functools import partial

class LocCaDataset(Dataset):
    def __init__(self, dataframe, tasks: list[str] = ["ARef Region","GCap", "ARef"], transforms= None):
        self.dataframe = dataframe
        self.transforms = transforms
        self.tasks = tasks
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index]["image_paths"]
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        sampled_task = random.choice(self.tasks)
        text = self.dataframe.iloc[index][sampled_task]
        return {"image": image, "text": sampled_task + ": " + text}


class RRGDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.dataframe = dataframe
        self.transforms=transforms
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = os.path.join("..", self.dataframe.iloc[index]["image_file_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        text = self.dataframe.iloc[index]['full_report']
        return {"image": image, "text": "Cap: " + text}


image_transforms = v2.Compose([
    v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # replaces ToTensor()
])


def test_collate_fn(examples, tokenizer):
    images = [example["image"] for example in examples]
    images = torch.stack(images)
    texts = [example["text"] for example in examples]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    )
    return {**inputs, "images": images}


def train_collate_fn(examples, tokenizer):
    images = [example["image"] for example in examples]
    images = torch.stack(images)
    texts = [example["text"] for example in examples]

    # Tokenize all texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )

    # Save a copy of the original input_ids as targets
    targets = inputs['input_ids'].clone()

    # Total number of examples
    batch_size = len(examples)

    # Randomly select 25% of indices to mask
    num_to_mask = int(batch_size * 0.25)
    indices_to_mask = random.sample(range(batch_size), num_to_mask)

    # Get PAD token ID
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Replace input ids and attention mask for selected examples
    for idx in indices_to_mask:
        seq_length = inputs["input_ids"].shape[1]
        inputs["input_ids"][idx] = torch.full((seq_length, ), pad_token_id, dtype=torch.long)
        inputs["attention_mask"][idx] = torch.zeros(seq_length, dtype=torch.long)

    # Return everything
    return {
        **inputs,
        "images": images,
        "targets": targets,
    }

def get_dataloader(dataset: Dataset, collate_fn, tokenizer, batch_size: int, train: bool=True):
    dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=8,
                        pin_memory=True,
                        prefetch_factor=4,
                        persistent_workers=True,
                        collate_fn=partial(collate_fn, 
                                        tokenizer=tokenizer)
                    )
    return dataloader
