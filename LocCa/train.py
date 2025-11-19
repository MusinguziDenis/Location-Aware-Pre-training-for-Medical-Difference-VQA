import torch
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    batch_bar = tqdm.tqdm(dataloader, desc="Train", total=len(dataloader), leave=False)
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        pixel_values = batch["images"].to(device, torch.bfloat16, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, bool, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, logits = model(
            tgt_input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            targets=targets,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        batch_bar.set_postfix(
            loss="{:.04f}".format((total_loss/(i+1)).cpu().item())
        )
        batch_bar.update()
    batch_bar.close()
    total_loss = total_loss / len(dataloader)
    return total_loss


@torch.no_grad()
def validation(model, dataloader):
    model.eval()
    total_loss = 0.0
    batch_bar = tqdm.tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        pixel_values = batch["images"].to(device, torch.bfloat16, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, bool, non_blocking=True)
        # targets = batch["targets"].to(device, non_blocking=True)
        loss, logits = model(
            tgt_input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            # targets=targets,
        )
        total_loss += loss.detach()
        batch_bar.set_postfix(
            loss="{:.04f}".format((total_loss/(i+1)).cpu().item())
        )
        batch_bar.update()
    batch_bar.close()
    total_loss = total_loss / len(dataloader)
    return total_loss
