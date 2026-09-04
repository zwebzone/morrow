from __future__ import annotations

import torch
import torch.nn.functional as F

from .backbone import GatedDeltaNetBackbone


def encode_prompt_and_completion(
    backbone: GatedDeltaNetBackbone,
    prompt: str,
    completion: str,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rendered = backbone.render_user(prompt, add_generation_prompt=True)
    prompt_ids = backbone.tokenizer(
        rendered, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    completion_ids = backbone.tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    eos = backbone.tokenizer.eos_token_id
    if eos is not None:
        completion_ids = torch.cat([completion_ids, torch.tensor([eos])])
    if completion_ids.numel() < 1:
        raise ValueError("completion produced no tokens")
    available_prompt = max(1, max_tokens - completion_ids.numel())
    prompt_ids = prompt_ids[-available_prompt:]
    input_ids = torch.cat([prompt_ids, completion_ids])[:max_tokens].unsqueeze(0)
    prompt_length = min(prompt_ids.numel(), input_ids.shape[1])
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    attention_mask = torch.ones_like(input_ids)
    device = backbone.device
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def completion_nll(
    backbone: GatedDeltaNetBackbone,
    prompt: str,
    completion: str,
    state: torch.Tensor | None,
    max_tokens: int,
) -> torch.Tensor:
    input_ids, attention_mask, labels = encode_prompt_and_completion(
        backbone, prompt, completion, max_tokens
    )
    outputs = backbone.forward_with_state(
        input_ids=input_ids,
        attention_mask=attention_mask,
        initial_state=state,
    )
    logits = outputs.logits[:, :-1].float()
    targets = labels[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def preservation_kl(
    backbone: GatedDeltaNetBackbone,
    prompt: str,
    completion: str,
    state: torch.Tensor,
    max_tokens: int,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    input_ids, attention_mask, labels = encode_prompt_and_completion(
        backbone, prompt, completion, max_tokens
    )
    with torch.no_grad():
        teacher = backbone.forward_with_state(
            input_ids=input_ids,
            attention_mask=attention_mask,
            initial_state=None,
        ).logits[:, :-1].float()
    student = backbone.forward_with_state(
        input_ids=input_ids,
        attention_mask=attention_mask,
        initial_state=state,
    ).logits[:, :-1].float()
    mask = labels[:, 1:] != -100
    if not torch.any(mask):
        raise ValueError("preservation example has no supervised completion tokens")
    teacher_log = F.log_softmax(teacher[mask] / temperature, dim=-1)
    student_log = F.log_softmax(student[mask] / temperature, dim=-1)
    teacher_probability = teacher_log.exp()
    token_kl = torch.sum(teacher_probability * (teacher_log - student_log), dim=-1)
    return token_kl.mean() * temperature**2

