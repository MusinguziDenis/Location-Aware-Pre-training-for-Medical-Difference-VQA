import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from dataclasses import dataclass, asdict
from PIL import Image
from data import image_transforms

from siglip import SiglipVisionModel, SiglipVisionConfig

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

@dataclass
class VisionEncoderConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 3072
    image_size: int =224
    patch_size: int =16


@dataclass
class DecoderConfig:
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    block_size: int = 1024
    dim_feedforward: int = 3072
    dropout: float = 0.1
    tokenizer: object = None
    vit_config: SiglipVisionConfig = SiglipVisionConfig

# Vision Transformer
class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vit_config = ViTConfig(**asdict(config))
        self.vit = ViTModel(vit_config)

    def forward(self, pixel_values):
        # output: [batch, num_patches +1, hidden_size]
        outputs = self.vit(pixel_values)
        return outputs.last_hidden_state


# Transformer Decoder with Cross-Attention
class TransformerDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.block_size, config.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            bias=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.fc_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.tokenizer.pad_token_id)


    def forward(self, tgt_input_ids, memory, tgt_is_causal=True, attention_mask=None, targets=None):
        # Embed target tokens and add positional encoding
        _, seq_len = tgt_input_ids.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.pos_embedding(pos) # position embeddings of shape (t, n_embd)
        tgt_emb = self.embedding(tgt_input_ids) + pos_emb
        causal_mask = torch.full(
            (seq_len, seq_len), fill_value=1, dtype=bool, device=device
        )
        causal_mask = torch.triu(causal_mask, 1).to(bool)
        attention_mask = ~attention_mask
        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask, tgt_key_padding_mask=attention_mask, tgt_is_causal=tgt_is_causal)
        logits= self.fc_out(out)
        shift_logits = logits[:, :-1].contiguous()
        if targets is not None:
            shift_labels = targets[:, 1:].contiguous()
        else:
            shift_labels = tgt_input_ids[:, 1:].contiguous()
        loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.vit_config.hidden_size, config.vit_config.projection_dim, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.vit_config.projection_dim, config.vit_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.gelu(self.linear_1(image_features))
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LocCaVLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = SiglipVisionModel(SiglipVisionConfig)
        self.text_decoder = TransformerDecoder(config)
        self.mm_projector = MultiModalProjector(config)

    def forward(self, pixel_values, tgt_input_ids, attention_mask, targets=None):
        image_features, pooler_output = self.vision_encoder(pixel_values)
        image_features = self.mm_projector(image_features)
        tgt_is_causal = False if attention_mask is None else True
        logits, loss = self.text_decoder(
            tgt_input_ids=tgt_input_ids,
            memory=image_features,
            attention_mask=attention_mask,
            targets=targets,
            tgt_is_causal=tgt_is_causal,
        )
        return loss, logits

    def tie_weights(self):
        self.text_decoder.tie_weights()

    def get_image_features(pixel_values):
        pass

    def get_text_features(text):
        pass
    
    @torch.no_grad()
    @torch.inference_mode()
    def generate(
            self,
            tokenizer,
            prompt: str,
            image_path: str,
            max_tokens_to_generate: int,
            temperature: float=0.3,
            top_k: int = 20,
            do_sample: bool = True,
        ):
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True
        )
        stop_token = tokenizer.eos_token_id
        generated_tokens = []
        image = Image.open(image_path).convert("RGB")
        image = image_transforms(image).unsqueeze(0)
        input_ids = inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(device, bool, non_blocking=True)

        for _ in range(max_tokens_to_generate):
            _, logits = self(
                pixel_values=image.to(device, torch.bfloat16, non_blocking=True),
                tgt_input_ids=input_ids,
                attention_mask=attention_mask,
            )
            next_token_logits = logits[:, -1, :]
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
            if next_token.item() == stop_token:
                break
            # Append the next token to the input
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device, dtype=bool)], dim=-1
            )
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded
