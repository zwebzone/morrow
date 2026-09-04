from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3.5-4B"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    write_tokens: int = 8
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(
        default_factory=lambda: ["all-linear"]
    )


@dataclass
class CarrierConfig:
    rank: int = 32
    interpolation: float = 0.75
    epsilon: float = 1e-6


@dataclass
class ObjectiveConfig:
    lambda_pre: float = 0.25
    lambda_preserve: float = 0.05
    current_qa_per_step: int = 2
    replay_qa_per_step: int = 2
    preservation_temperature: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 42
    steps: int = 20_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    max_unroll: int = 10
    curriculum_steps: int = 10_000
    tbptt_horizon: int = 2
    max_context_tokens: int = 2048
    max_query_tokens: int = 512
    log_every: int = 10
    save_every: int = 1_000


@dataclass
class DataConfig:
    session_jsonl: str = "data/train_sessions.jsonl"
    capability_jsonl: str = "data/capability.jsonl"


@dataclass
class OutputConfig:
    directory: str = "outputs/morrow"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.model.write_tokens < 1:
            raise ValueError("model.write_tokens must be positive")
        if self.model.lora_rank < 1 or self.carrier.rank < 1:
            raise ValueError("all low-rank dimensions must be positive")
        if not 0.0 <= self.carrier.interpolation <= 1.0:
            raise ValueError("carrier.interpolation must be in [0, 1]")
        if self.training.max_unroll < 1:
            raise ValueError("training.max_unroll must be positive")
        if self.training.tbptt_horizon < 1:
            raise ValueError("training.tbptt_horizon must be positive")
        if self.training.tbptt_horizon > self.training.max_unroll:
            raise ValueError("tbptt_horizon cannot exceed max_unroll")
        if self.objective.current_qa_per_step < 1:
            raise ValueError("at least one current-session QA is required")
        if self.objective.lambda_pre < 0 or self.objective.lambda_preserve < 0:
            raise ValueError("loss weights must be non-negative")


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"model", "carrier", "objective", "training", "data", "output"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    cfg = ExperimentConfig()
    for section in allowed:
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section] or {})
    cfg.validate()
    return cfg
