import os
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import ConcatDataset
import torch

from data import LocCaDataset, get_dataloader, image_transforms

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# create training configuration
config = dict(
    lr=1e-4,
    batch_size=32,
    vision_encoder="SiglipVision",
    text_decoder="TransformerDecoder",
    epochs=2,
    dataset="combined-dataset",
    parallel_ratio=0.25,
    image_size=224,
)

# Create tokenizer
# Create Tokenizer
from transformers import GPT2Tokenizer

tokenizer = tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
locca_dataset = pd.read_csv("path-to-dataset")
locca_dataset.sample()

# Split the dataset
train_df, test_df = train_test_split(locca_dataset, test_size=0.05, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.05, random_state=42)

# Create datasets
train_dataset = LocCaDataset(
    train_df,
    transforms=image_transforms,
)

valid_dataset = LocCaDataset(
    valid_df,
    transforms=image_transforms,
)

test_dataset = LocCaDataset(
    test_df,
    transforms=image_transforms,
)

# Dataloader
from data import train_collate_fn, test_collate_fn

train_dataloader = get_dataloader(
    dataset=train_dataset,
    collate_fn=train_collate_fn,
    tokenizer=tokenizer,
    train=True,
    batch_size=config["batch_size"],
)

valid_dataloader = get_dataloader(
    dataset=valid_dataset,
    collate_fn=test_collate_fn,
    tokenizer=tokenizer,
    train=False,
    batch_size=config["batch_size"],
)

test_dataloader = get_dataloader(
    dataset=test_dataset,
    collate_fn=test_collate_fn,
    tokenizer=tokenizer,
    train=False,
    batch_size=config["batch_size"],
)

# Model
from model import LocCaVLM, DecoderConfig

decoder_config = DecoderConfig(tokenizer=tokenizer)
model = LocCaVLM(decoder_config)

model = model.to(torch.bfloat16).to(device)

torch.compile(model)

# Train & Validation
from train import train, validation

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)

# Weights & Biases
import wandb

wandb.login()

run = wandb.init(
    name=f"vision_encoder_{config['vision_encoder']}_text_decoder_{config['text_decoder']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}_dataset_{config['dataset']}_parallel_ratio_{config['parallel_ratio']}_image_res_{config['image_size']}",
    reinit=True,
    project="",
    config=config,
)

# setup wandb watch
wandb.watch(model, log="all", log_freq=len(train_dataloader))

# Experiment
best_val_loss = float("Inf")
for epoch in range(config['epochs']):
    curr_lr = float(optimizer.param_groups[0]['lr'])
    train_loss  = train(model, train_dataloader, optimizer)
    scheduler.step()
    valid_loss = validation(model=model, dataloader=valid_dataloader)
    print(f"Epoch: {epoch+1} \t Train Loss: {train_loss:.04f} \t Learning rate: {curr_lr:.06f}")
    print(f"Validation Loss: {valid_loss:.04f}")
    wandb.log({
        "Train/Epoch": epoch + 1,
        "Train/Train_loss": train_loss,
        "Val/Valid_loss": valid_loss,
        "Train/Lr": curr_lr
    })
    if valid_loss < best_val_loss:
        if not os.path.exists("./model"):
            os.makedirs("./model")
        best_val_loss = valid_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "Epoch" : epoch + 1
        },
        f"./model/vision_encoder_{config['vision_encoder']}_text_decoder_{config['text_decoder']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}_dataset_{config['dataset']}_parallel_ratio_{config['parallel_ratio']}_image_res_{config['image_size']}.pth"
        )
    response = model.generate(
        tokenizer=tokenizer,
        prompt="GCap: ",
        image_path="",
        temperature=0.3,
        top_k=20,
        do_sample=True,
        max_tokens_to_generate=50,
    )
    print(response)

run.finish()

