from dataclasses import dataclass 
import math
import warnings
from collections.abc import Callable
from typing import Any, Optional, Union, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class SiglipTextConfig:
    vocab_size: int = 32000
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    projection_size: int = 1152
    _attn_implementation: str = "flash"


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int =12
    num_channels: int = 3
    image_size: int = 224 # 448
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    projection_dim: int = 1280
    vision_use_head: bool = False


class SiglipTextModelOutput:
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches = (self.image_size//self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids) # [:, :embeddings.shape[1]]
        return embeddings


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else input_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if input_embeds is None:
            input_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = input_embeds + position_embeddings

        return embeddings

def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SiglipAttention(nn.Module):
    """Multi-headed attantion from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor]=None,
            **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling =self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: Union[SiglipTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_embeds,
        attention_mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return hidden_states


class SiglipTextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # Siglip text model does not use a causal mask
        uses_flash_attention = "flash" in self.config._attn_implementation
        if uses_flash_attention:
            attention_mask = None
        elif attention_mask is not None and not uses_flash_attention:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # the models use the last token's hidden state, which may be padding
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return hidden_states, pooled_output


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(input_embeds=hidden_states)
        last_hidden_states = self.post_layernorm(last_hidden_states)
        pooler_output = self.head(last_hidden_states) if self.use_head else None

        return last_hidden_states, pooler_output

    
class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """ Multihead Attention Pooling"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_states = self.attention(probe, hidden_states, hidden_states)[0]

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states[:, 0]


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)


class SiglipTextModel(nn.Module):
    config: SiglipTextConfig

    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.text_model = SiglipTextTransformer(config)
        
    def get_input_enbeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, SiglipTextModel

        >>> model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        """

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class SiglipModel(nn.Module):
    def __init__(self, text_config: SiglipTextConfig, vision_config: SiglipVisionConfig):
        super().__init__()
        text_model = SiglipTextModel(text_config)
        vision_model = SiglipVisionModel(vision_config)

        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.log(torch.ones((1))*10))
        self.logit_bias  = nn.Parameter(torch.ones((1))*-10)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return text_outputs
    
    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
    ) -> torch.Tensor:
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        return vision_outputs
    

    def forward(
            self,
            input_ids: Optional[torch.LongTensor],
            pixel_values: Optional[torch.FloatTensor],
            attention_mask: Optional[torch.FloatTensor]=None,
            position_ids: Optional[torch.LongTensor]=None,
    ):
        _, image_embeds = self.vision_model(
            pixel_values=pixel_values,
        )
        _, text_embeds = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logits_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logits_bias

        logits_per_image = logits_per_text.t()

        eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
        m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        loglik = nn.functional.logsigmoid(m1_diag1 * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()

        return loss

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in ["google/medsiglip-448"], f"The model you stated {model_name} is not supported."
        from transformers import AutoModel
        print("loading weights from pretrained: %s" %model_name)

        text_config = SiglipTextConfig()
        vision_config = SiglipVisionConfig()

        model = SiglipModel(text_config=text_config, vision_config=vision_config)
        hf_model = AutoModel.from_pretrained(model_name)
        sd_hf_model = hf_model.state_dict()

        model.load_state_dict(sd_hf_model)

        return model