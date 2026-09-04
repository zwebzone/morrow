from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def transformer_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        ("model", "model", "layers"),
        ("model", "layers"),
    )
    for path in candidates:
        node = model
        try:
            for part in path:
                node = getattr(node, part)
        except AttributeError:
            continue
        if isinstance(node, nn.ModuleList):
            return node
    raise ValueError("unable to locate transformer layers in the supplied model")


class GatedDeltaNetBackbone(nn.Module):
    """Cold-KV wrapper exposing native recurrent states of hybrid GatedDeltaNet LMs."""

    def __init__(self, name_or_path: str, dtype: str, trust_remote_code: bool):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=resolve_dtype(dtype),
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.requires_grad_(False)
        self.model.eval()

        layer_types = getattr(getattr(self.model.config, "text_config", self.model.config), "layer_types", None)
        layers = transformer_layers(self.model)
        if layer_types is None or len(layer_types) != len(layers):
            raise ValueError("model configuration must expose one layer_type per transformer layer")
        self.recurrent_model_indices = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
        ]
        if not self.recurrent_model_indices:
            raise ValueError("the model contains no linear_attention recurrent layers")
        self.recurrent_layers = [layers[index] for index in self.recurrent_model_indices]
        for layer in self.recurrent_layers:
            if not hasattr(layer, "linear_attn") or not hasattr(
                layer.linear_attn, "chunk_gated_delta_rule"
            ):
                raise ValueError("linear_attention layer does not expose a GatedDeltaNet kernel")

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def hidden_size(self) -> int:
        config = getattr(self.model.config, "text_config", self.model.config)
        return int(config.hidden_size)

    @contextmanager
    def initial_state(self, state: torch.Tensor | None) -> Iterator[None]:
        if state is None:
            yield
            return
        if state.ndim != 4 or state.shape[0] != len(self.recurrent_layers):
            raise ValueError(
                "state must have shape [recurrent_layers, heads, key_dim, value_dim]"
            )
        originals: list[tuple[nn.Module, object]] = []

        def make_wrapper(original: object, layer_state: torch.Tensor):
            def wrapped(*args, **kwargs):
                reference = args[0] if args and isinstance(args[0], torch.Tensor) else layer_state
                kwargs["initial_state"] = layer_state.to(
                    device=reference.device, dtype=torch.float32
                ).unsqueeze(0).expand(reference.shape[0], -1, -1, -1).contiguous()
                return original(*args, **kwargs)

            return wrapped

        try:
            for recurrent_index, layer in enumerate(self.recurrent_layers):
                owner = layer.linear_attn
                original = owner.chunk_gated_delta_rule
                originals.append((owner, original))
                owner.chunk_gated_delta_rule = make_wrapper(original, state[recurrent_index])
            yield
        finally:
            for owner, original in originals:
                owner.chunk_gated_delta_rule = original

    def forward_with_state(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        initial_state: torch.Tensor | None,
    ):
        with self.initial_state(initial_state):
            return self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

    def terminal_recurrent_state(self, outputs) -> torch.Tensor:
        cache = outputs.past_key_values
        states: list[torch.Tensor] = []
        for model_index in self.recurrent_model_indices:
            recurrent = getattr(cache, "recurrent_states", None)
            if recurrent is not None:
                state = recurrent[model_index]
            else:
                cache_layers = getattr(cache, "layers", None)
                if cache_layers is None:
                    raise ValueError("model output cache does not expose recurrent_states")
                state = getattr(cache_layers[model_index], "recurrent_states", None)
            if state is None:
                raise ValueError(f"missing recurrent state for model layer {model_index}")
            if state.ndim != 4 or state.shape[0] != 1:
                raise ValueError("reference trainer requires batch size one recurrent states")
            states.append(state[0])
        return torch.stack(states, dim=0)

    def render_user(self, content: str, add_generation_prompt: bool) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        suffix = "\nAnswer:" if add_generation_prompt else ""
        return f"{content}{suffix}"
