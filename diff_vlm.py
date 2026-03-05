"""
Implementation of the differential multimodal model
"""

from gpt import GPT
from LocCa.siglip import SiglipVisionModel, SiglipVisionConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from PIL import Image
from torchvision.transforms import transforms
import inspect
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class VisionConfig:
    model: str = "google/medsiglip-448"
    hidden_size: int = 1152
    projection_dim: int = 1024
    num_image_slots: int = 2

@dataclass
class VLMConfig:
    vision_config: VisionConfig = VisionConfig
    vocab_size: int = 50258
    pad_token_id: int = 50257
    model_type: str = "gpt2-medium"


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.vision_config.projection_dim, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.gelu(self.linear_1(image_features))
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class DiffVLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Vision
        self.vision_encoder = SiglipVisionModel(SiglipVisionConfig)
        self.image_id_embeddings = nn.Embedding(config.vision_config.num_image_slots, config.vision_config.hidden_size)
        # mm connector
        self.mm_projector = MultiModalProjector(config)
        # language model
        self.vocab_size = config.vocab_size
        language_model = GPT.from_pretrained(config.model_type)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()  # implement tie weights in language model

    def _merge_input_ids_with_image_features(
            self, ref_image_features: torch.Tensor, main_image_features: torch.Tensor, input_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, mm_token_type_ids: torch.Tensor,
    ):
        batch_size, _, embed_dim = input_embeds.shape
        b, s = input_ids.shape  # comment out later
        dtype, device = input_embeds.dtype, input_embeds.device

        # concatenate main and reference image features along seq length
        ref_image_features = self.mm_projector(ref_image_features)
        main_image_features = self.mm_projector(main_image_features)
        combined_image_features = torch.cat((ref_image_features, main_image_features), dim=1)
        # project the image feature dimension to that of the language model
        # embedded text includes the embeddings of the image token, fill these tokens with image tokens
        image_mask_expanded = mm_token_type_ids.unsqueeze(-1).expand(-1, -1, embed_dim).to(bool)
        input_embeds = input_embeds.masked_scatter(image_mask_expanded, combined_image_features)

        ### Create the attention mask
        dtype, device = input_embeds.dtype, input_embeds.device
        # min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        causal_mask = torch.full(
            (batch_size, q_len, q_len), fill_value=1, dtype=bool, device=device
        )
        causal_mask = torch.triu(causal_mask, 1).unsqueeze(1).to(bool)
        # Add the head dimension
        # Add head and seq len dim to the mask mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(bool)
        combined_mask = ~attention_mask | causal_mask

        return input_embeds, ~combined_mask
    
    def _add_image_id_embeddings(self, image_embeds, image_index):
        b, n, d = image_embeds.shape
        id_embed = self.image_id_embeddings(torch.tensor(image_index, device=image_embeds.device))
        id_embed = id_embed.unsqueeze(0).unsqueeze(1).expand(b, n, d)
        return image_embeds + id_embed

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        main_pixel_values: torch.FloatTensor = None,
        ref_pixel_values: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        mm_token_type_ids: torch.LongTensor = None,
        targets: torch.LongTensor = None,
    ) -> Tuple:
        batch_size = main_pixel_values.shape[0]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)  # implement get_input_embeddings

        # stack main and reference images and perform a single forward pass
        stacked_pixel_values = torch.cat((main_pixel_values, ref_pixel_values), dim=0)
        stacked_image_features, _ = self.vision_encoder(stacked_pixel_values.to(input_embeds.dtype)) # removed the call to forward features removed last_hidden_state for new implementation of Siglip
        main_image_features, ref_image_features = torch.split(stacked_image_features, [batch_size, batch_size], dim=0)
        # Merge text and images
        # Add image id embeds to the image features
        main_image_features = self._add_image_id_embeddings(main_image_features, 1)
        ref_image_features = self._add_image_id_embeddings(ref_image_features, 0)
        input_embeds, attention_mask = self._merge_input_ids_with_image_features(
            ref_image_features,
            main_image_features,
            input_embeds,
            input_ids,
            attention_mask,
            mm_token_type_ids,
        )
        outputs = self.language_model(
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            targets=targets,
        )
        return outputs

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter parameters that require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 ]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(param.numel() for param in decay_params)
        num_non_decay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_non_decay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer
    
    @staticmethod
    def get_model_inputs(
        processor,
        main_image_path,
        ref_image_path,
        prompt,
    ):
        transformation = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])
        main_image = Image.open(main_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
        main_images = [transformation(main_image)]
        ref_images = [transformation(ref_image)]
        prompts = [prompt]
        model_inputs = processor(
            main_images=main_images,
            ref_images=ref_images,
            text=prompts,
            suffix=None,
        )
        return model_inputs

    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        self,
        processor,
        prompt: str,
        main_image_path: str,
        ref_image_path: str,
        max_tokens_to_generate: int=64,
        temperature: float=0.1,
        top_k: int=20,
        do_sample: bool=True,
    ):
        model_inputs = DiffVLM.get_model_inputs(
            processor=processor,
            main_image_path=main_image_path,
            ref_image_path=ref_image_path,
            prompt=prompt,
        )
        input_ids = model_inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = model_inputs["attention_mask"].to(device, non_blocking=True)
        main_pixel_values = model_inputs["pixel_values_main"].to(device, non_blocking=True)
        ref_pixel_values = model_inputs["pixel_values_ref"].to(device, non_blocking=True)
        mm_token_type_ids = model_inputs["mm_token_type_ids"].to(device, non_blocking=True)

        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []

        for _ in range(max_tokens_to_generate):
            outputs = self(
                input_ids=input_ids,
                main_pixel_values=main_pixel_values,
                ref_pixel_values=ref_pixel_values,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids
            )
            next_token_logits = outputs["logits"][:, -1, :]
            if top_k is not None:
                # Apply temperature
                next_token_logits = next_token_logits /temperature
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1))) 
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            # stop if the stop token has been generated
            if next_token.item() == stop_token:
                break
            # Append the next token to the input
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )
            # Append 0 to the image mask
            mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros((1, 1), device=input_ids.device)], dim=-1)
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return decoded
