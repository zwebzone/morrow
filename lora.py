from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass
class AdapterRuntime:
    enabled: bool = False


class LoRAResidual(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        adapter_input = self.dropout(hidden).to(self.down.weight.dtype)
        return self.up(self.down(adapter_input)) * self.scale


class LinearWithWriteLoRA(nn.Module):
    """Frozen base projection plus a residual enabled only during memory writing."""

    def __init__(self, base: nn.Linear, adapter: LoRAResidual, runtime: AdapterRuntime):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_runtime", runtime)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden)
        if self._runtime.enabled:
            output = output + self._adapter(hidden).to(output.dtype)
        return output


def _module_and_leaf(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent = root
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


class WriteInterface(nn.Module):
    """Learned write tokens and write-only low-rank backbone adapters."""

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        write_tokens: int,
        rank: int,
        alpha: float,
        dropout: float,
        targets: list[str],
    ):
        super().__init__()
        self.runtime = AdapterRuntime()
        self.write_tokens = nn.Parameter(torch.empty(write_tokens, hidden_size))
        nn.init.normal_(self.write_tokens, std=0.02)
        self.adapters = nn.ModuleDict()

        selected: list[tuple[str, nn.Linear]] = []
        for path, module in model.named_modules():
            if not isinstance(module, nn.Linear) or path.endswith("lm_head"):
                continue
            leaf = path.rsplit(".", 1)[-1]
            if "all-linear" in targets or leaf in targets or path in targets:
                selected.append((path, module))
        if not selected:
            raise ValueError(f"no linear modules matched LoRA targets: {targets}")

        for index, (path, base) in enumerate(selected):
            key = f"adapter_{index:04d}"
            adapter = LoRAResidual(
                input_dim=base.in_features,
                output_dim=base.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.adapters[key] = adapter
            parent, leaf = _module_and_leaf(model, path)
            setattr(parent, leaf, LinearWithWriteLoRA(base, adapter, self.runtime))

    @contextmanager
    def writing(self) -> Iterator[None]:
        previous = self.runtime.enabled
        self.runtime.enabled = True
        try:
            yield
        finally:
            self.runtime.enabled = previous

    @contextmanager
    def reading(self) -> Iterator[None]:
        """Force the write adapter off for answering and preservation."""
        previous = self.runtime.enabled
        self.runtime.enabled = False
        try:
            yield
        finally:
            self.runtime.enabled = previous

    def append_write_tokens(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = token_embeddings.shape[0]
        tokens = self.write_tokens.to(
            device=token_embeddings.device, dtype=token_embeddings.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
        embeddings = torch.cat([token_embeddings, tokens], dim=1)
        token_mask = torch.ones(
            batch,
            tokens.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        return embeddings, torch.cat([attention_mask, token_mask], dim=1)
