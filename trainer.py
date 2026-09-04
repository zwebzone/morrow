from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .backbone import GatedDeltaNetBackbone
from .carrier import LowRankStateCarrier, stable_slerp_carry
from .config import ExperimentConfig
from .data import CapabilityExample, QA, SequenceSampler, SessionSequence
from .lora import WriteInterface
from .objectives import completion_nll, preservation_kl


def curriculum_depth(step: int, curriculum_steps: int, maximum_depth: int) -> int:
    if maximum_depth < 1:
        raise ValueError("maximum_depth must be positive")
    if curriculum_steps <= 0:
        return maximum_depth
    progress = min(max(step, 0) / curriculum_steps, 1.0)
    return 1 + int(progress * (maximum_depth - 1))


class MorrowModel(nn.Module):
    def __init__(
        self,
        backbone: GatedDeltaNetBackbone,
        writer: WriteInterface,
        carrier: LowRankStateCarrier,
        interpolation: float,
        epsilon: float,
        max_context_tokens: int,
        max_query_tokens: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.writer = writer
        self.carrier = carrier
        self.interpolation = interpolation
        self.epsilon = epsilon
        self.max_context_tokens = max_context_tokens
        self.max_query_tokens = max_query_tokens

    def write(
        self, context: str, previous_state: torch.Tensor | None
    ) -> torch.Tensor:
        rendered = self.backbone.render_user(context, add_generation_prompt=False)
        encoded = self.backbone.tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_context_tokens,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.backbone.device)
        attention_mask = encoded.attention_mask.to(self.backbone.device)
        embeddings = self.backbone.model.get_input_embeddings()(input_ids)
        embeddings, attention_mask = self.writer.append_write_tokens(
            embeddings, attention_mask
        )
        with self.writer.writing():
            outputs = self.backbone.forward_with_state(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                initial_state=previous_state,
            )
        terminal = self.backbone.terminal_recurrent_state(outputs)
        candidate = self.carrier(terminal)
        if previous_state is None:
            return candidate
        return stable_slerp_carry(
            previous_state,
            candidate,
            interpolation=self.interpolation,
            epsilon=self.epsilon,
        )

    def post_transition_loss(self, qa: QA, state: torch.Tensor) -> torch.Tensor:
        with self.writer.reading():
            return completion_nll(
                self.backbone,
                prompt=qa.question,
                completion=qa.answer,
                state=state,
                max_tokens=self.max_query_tokens,
            )

    def pre_transition_loss(
        self, context: str, qa: QA, previous_state: torch.Tensor
    ) -> torch.Tensor:
        prompt = f"Context:\n{context}\n\nQuestion:\n{qa.question}"
        with self.writer.reading():
            return completion_nll(
                self.backbone,
                prompt=prompt,
                completion=qa.answer,
                state=previous_state,
                max_tokens=self.max_query_tokens,
            )

    def preservation_loss(
        self,
        example: CapabilityExample,
        state: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        with self.writer.reading():
            return preservation_kl(
                self.backbone,
                prompt=example.prompt,
                completion=example.completion,
                state=state,
                max_tokens=self.max_query_tokens,
                temperature=temperature,
            )

    def public_state_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for prefix, module in (("writer", self.writer), ("carrier", self.carrier)):
            for name, tensor in module.state_dict().items():
                result[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
        return result

    def public_state_keys(self) -> set[str]:
        return {
            f"{prefix}.{name}"
            for prefix, module in (("writer", self.writer), ("carrier", self.carrier))
            for name in module.state_dict()
        }

    def load_public_checkpoint(self, path: str | Path) -> None:
        tensors = load_file(str(path), device="cpu")
        expected = self.public_state_keys()
        received = set(tensors)
        if expected != received:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            raise ValueError(
                f"checkpoint contract mismatch; missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        writer = {
            key.removeprefix("writer."): value
            for key, value in tensors.items()
            if key.startswith("writer.")
        }
        carrier = {
            key.removeprefix("carrier."): value
            for key, value in tensors.items()
            if key.startswith("carrier.")
        }
        self.writer.load_state_dict(writer, strict=True)
        self.carrier.load_state_dict(carrier, strict=True)


def build_model(config: ExperimentConfig, device: torch.device) -> MorrowModel:
    backbone = GatedDeltaNetBackbone(
        config.model.name_or_path,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    backbone.model.to(device)
    writer = WriteInterface(
        backbone.model,
        hidden_size=backbone.hidden_size,
        write_tokens=config.model.write_tokens,
        rank=config.model.lora_rank,
        alpha=config.model.lora_alpha,
        dropout=config.model.lora_dropout,
        targets=config.model.lora_targets,
    ).to(device)

    with torch.no_grad(), writer.writing():
        encoded = backbone.tokenizer(
            "Initialize recurrent-state dimensions.", return_tensors="pt"
        ).to(device)
        embeddings = backbone.model.get_input_embeddings()(encoded.input_ids)
        embeddings, attention_mask = writer.append_write_tokens(
            embeddings, encoded.attention_mask
        )
        output = backbone.forward_with_state(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            initial_state=None,
        )
        state = backbone.terminal_recurrent_state(output)
    layers, heads, key_dim, value_dim = state.shape
    carrier = LowRankStateCarrier(
        layers=layers,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        rank=config.carrier.rank,
    ).to(device)
    return MorrowModel(
        backbone=backbone,
        writer=writer,
        carrier=carrier,
        interpolation=config.carrier.interpolation,
        epsilon=config.carrier.epsilon,
        max_context_tokens=config.training.max_context_tokens,
        max_query_tokens=config.training.max_query_tokens,
    )


class MorrowTrainer:
    def __init__(
        self,
        model: MorrowModel,
        config: ExperimentConfig,
        sequences: list[SessionSequence],
        capability_examples: list[CapabilityExample],
    ):
        self.model = model
        self.config = config
        self.sampler = SequenceSampler(sequences, config.training.seed)
        self.capability_examples = capability_examples
        self.random = random.Random(config.training.seed + 1)
        sequence_sources = {item.sequence_id for item in sequences}
        capability_sources = {item.source_id for item in capability_examples}
        overlap = sequence_sources & capability_sources
        if overlap:
            raise ValueError(
                f"capability data must be source-disjoint from memory data; overlap: {sorted(overlap)[:5]}"
            )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("no trainable Morrow parameters found")
        if any(parameter.requires_grad for parameter in model.backbone.model.parameters()):
            raise ValueError("pretrained backbone must remain frozen")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def _qa_buffer(self, sequence: SessionSequence, depth: int) -> list[QA]:
        current = sequence.sessions[depth - 1].qa
        history = [qa for session in sequence.sessions[: depth - 1] for qa in session.qa]
        return self.sampler.sample_items(
            current, self.config.objective.current_qa_per_step
        ) + self.sampler.sample_items(
            history, self.config.objective.replay_qa_per_step
        )

    def _unroll(
        self, sequence: SessionSequence, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state: torch.Tensor | None = None
        previous_for_terminal: torch.Tensor | None = None
        gradient_start = max(0, depth - self.config.training.tbptt_horizon)
        for index, session in enumerate(sequence.sessions[:depth]):
            if index == depth - 1:
                previous_for_terminal = state
            if index < gradient_start:
                with torch.no_grad():
                    state = self.model.write(session.context, state)
                state = state.detach()
            else:
                state = self.model.write(session.context, state)
        if state is None:
            raise RuntimeError("unroll produced no persistent state")
        return state, previous_for_terminal

    def step(self, step: int) -> dict[str, float | int | str]:
        maximum_depth = curriculum_depth(
            step,
            self.config.training.curriculum_steps,
            self.config.training.max_unroll,
        )
        sequence, depth = self.sampler.sample(maximum_depth)
        state, previous = self._unroll(sequence, depth)
        qa_buffer = self._qa_buffer(sequence, depth)
        post = torch.stack(
            [self.model.post_transition_loss(qa, state) for qa in qa_buffer]
        ).mean()

        pre = post.new_zeros(())
        if depth > 1:
            if previous is None:
                raise RuntimeError("depth > 1 requires a previous persistent state")
            current_context = sequence.sessions[depth - 1].context
            pre = torch.stack(
                [
                    self.model.pre_transition_loss(current_context, qa, previous)
                    for qa in qa_buffer
                ]
            ).mean()

        preserve_example = self.random.choice(self.capability_examples)
        preserve = self.model.preservation_loss(
            preserve_example,
            state,
            temperature=self.config.objective.preservation_temperature,
        )
        total = (
            post
            + self.config.objective.lambda_pre * pre
            + self.config.objective.lambda_preserve * preserve
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.training.gradient_clip,
        )
        self.optimizer.step()
        return {
            "step": step,
            "sequence_id": sequence.sequence_id,
            "depth": depth,
            "curriculum_max_depth": maximum_depth,
            "loss": float(total.detach()),
            "loss_post": float(post.detach()),
            "loss_pre": float(pre.detach()),
            "loss_preserve": float(preserve.detach()),
            "gradient_norm": float(gradient_norm),
            "state_norm": float(torch.linalg.vector_norm(state.detach())),
        }

    def save(self, output: Path, step: int) -> None:
        checkpoint = output / f"checkpoint-{step:08d}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        save_file(self.model.public_state_dict(), checkpoint / "morrow.safetensors")
        metadata = {"step": step, "configuration": asdict(self.config)}
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def train(self) -> None:
        output = Path(self.config.output.directory)
        output.mkdir(parents=True, exist_ok=True)
        metrics_path = output / "metrics.jsonl"
        self.model.backbone.model.eval()
        self.model.writer.train()
        self.model.carrier.train()
        with metrics_path.open("a", encoding="utf-8") as metrics:
            for step in range(1, self.config.training.steps + 1):
                values = self.step(step)
                metrics.write(json.dumps(values, sort_keys=True) + "\n")
                metrics.flush()
                if step % self.config.training.log_every == 0:
                    print(json.dumps(values, sort_keys=True), flush=True)
                if step % self.config.training.save_every == 0:
                    self.save(output, step)
        (output / "training.complete").write_text("complete\n", encoding="utf-8")
