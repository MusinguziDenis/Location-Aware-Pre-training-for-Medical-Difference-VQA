import torch.nn as nn
import torch

import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vlm_train(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    batch_bar = tqdm.tqdm(dataloader, desc="Train", total=len(dataloader), leave=False)
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        main_pixel_values = batch["pixel_values_main"].to(device, non_blocking=True)
        ref_pixel_values = batch["pixel_values_ref"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        mm_token_type_ids = batch["mm_token_type_ids"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=input_ids,
            main_pixel_values=main_pixel_values,
            ref_pixel_values=ref_pixel_values,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            targets=targets,
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss/(i+1))),
        )
        batch_bar.update()

    batch_bar.close()
    total_loss = total_loss / len(dataloader)

    return total_loss


@torch.no_grad()
@torch.inference_mode()
def vlm_validation(model, dataloader):
    model.eval()
    total_loss = 0
    batch_bar = tqdm.tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        main_pixel_values = batch["pixel_values_main"].to(device, non_blocking=True)
        ref_pixel_values = batch["pixel_values_ref"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        mm_token_type_ids = batch["mm_token_type_ids"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                main_pixel_values=main_pixel_values,
                ref_pixel_values=ref_pixel_values,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                targets=targets,
            )
        loss = output["loss"]
        total_loss += loss.item()
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss/(i+1))),
        )
        batch_bar.update()

    batch_bar.close()
    total_loss = total_loss / len(dataloader)

    return total_loss