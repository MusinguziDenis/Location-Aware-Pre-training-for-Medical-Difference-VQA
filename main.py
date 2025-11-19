"""Running the DiffVLM model for medical difference VQA."""

import torch
from vlm_processor import Processor
import os
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Config
config = {
    "lr": 2e-4,
    "epochs": 2,
    "batch_size": 16,
    "language_model": "gpt2-medium",
    "vision_encoder": "locca_vision_encoder",
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "image_seq_length": 256,
    "image_size": 224,
}

# Data
from data import VLMDataset, get_dataloader, train_transforms, test_transforms, collate_function, test_collate_function
from diff_vlm import VisionConfig

# Load dataset
df = pd.read_csv("p10_p12_diff_dataset.csv")

train_valid_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.1, random_state=42)

# Create datasets
train_dataset = VLMDataset(dataframe=train_df, transforms=train_transforms)
valid_dataset = VLMDataset(dataframe=valid_df, transforms=test_transforms)
test_dataset = VLMDataset(dataframe=test_df, transforms=test_transforms)

# Create processor
# def image_processor(images):
#     return torch.stack(images)

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(VisionConfig.model)
image_processor = processor.image_processor
image_processor.do_rescale = False
image_processor.size = {
                    "height": config['image_size'],
                    "width": config['image_size'],
                }

# Create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create VLM processor
vlm_processors = Processor(
    image_processor=image_processor,
    tokenizer=tokenizer,
    image_seq_length=config['image_seq_length']
)

train_dataloader = get_dataloader(
    dataset=train_dataset,
    collate_fn=collate_function,
    processor=vlm_processors,
    train=True,
    batch_size=config["batch_size"],
)

valid_dataloader = get_dataloader(
    dataset=valid_dataset,
    collate_fn=collate_function,
    processor=vlm_processors,
    train=False,
    batch_size=config["batch_size"],
)

test_dataloader = get_dataloader(
    dataset=test_dataset,
    collate_fn=test_collate_function,
    processor=vlm_processors,
    train=False,
    batch_size=config["batch_size"],
)

# Model
from diff_vlm import DiffVLM, VLMConfig

vlm = DiffVLM(VLMConfig)

# Load vision encoder checkpoint
locca_vision_encoder_sd = torch.load("", map_location="cpu")["model_state_dict"]

for name, param in vlm.named_parameters():
    if name.startswith("vision_encoder"):
        src_param = locca_vision_encoder_sd.get(name, None)
        if src_param is not None:
            if param.shape == src_param.shape:
                with torch.no_grad():
                    param.copy_(src_param)
            else:
                print(f"Skipping {name}: shape mismatch {param.shape} vs {src_param.shape}")

vlm = vlm.to(torch.bfloat16).to(device)

# Freeze the image encoder
for pn, p in vlm.named_parameters():
    if pn.startswith("vision_encoder."):
        p.requires_grad = False

# Train & Validation
from train import vlm_train, vlm_validation

optimizer = vlm.configure_optimizers(config["weight_decay"], config["lr"], betas=(config["beta1"], config["beta2"]), device_type="cuda") # config["lr"]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * config["epochs"], eta_min=1e-6)

# Weights & Biases
import wandb

wandb.login()

run = wandb.init(
    name = f"Diff_language_model_{config['language_model']}_vision_encoder_{config['vision_encoder'].replace('/', '_')}_pretrained_lr_{config['lr']}_bs_{config['batch_size']}_epochs_{config['epochs']}_image_size_{config['image_size']}",
    project="Medical-Diff-VQA",
    reinit=True,
    config=config,
)

wandb.watch(vlm, log="all", log_freq=len(train_dataloader))

# Experiment
best_val_loss = float("Inf")
for epoch in range(config['epochs']):
    curr_lr = float(optimizer.param_groups[0]['lr'])
    train_loss  = vlm_train(vlm, train_dataloader, optimizer, scheduler)
    valid_loss = vlm_validation(model=vlm, dataloader=valid_dataloader)
    print(f"Epoch: {epoch+1} \t Train Loss: {train_loss:.04f} \t Learning rate: {curr_lr:.06f}")
    print(f"Validation Loss: {valid_loss:.04f}")
    wandb.log({
        "Train/Epoch": epoch + 1,
        "Train/Loss": train_loss,
        "Val/Loss": valid_loss,
        "Train/Lr": curr_lr
    })
    if valid_loss < best_val_loss:
        if not os.path.exists("./model"):
            os.makedirs("./model")
        best_val_loss = valid_loss
        torch.save({
            "model_state_dict": vlm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "Epoch" : epoch + 1
        },
        f"./model/Diff_language_model_{config['language_model']}_vision_encoder_{config['vision_encoder'].replace('/', '_')}_pretrained_lr_{config['lr']}_bs_{config['batch_size']}_epochs_{config['epochs']}_image_size_{config['image_size']}.pth"
        )
    response = vlm.generate(
        processor=vlm_processors,
        prompt="what has changed compared to the reference image?",
        main_image_path="",
        ref_image_path="",
        temperature=0.1,
        top_k=40,
        do_sample=False,
        max_tokens_to_generate=50,
    )
    print(response)

run.finish()