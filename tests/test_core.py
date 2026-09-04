import json

import pytest
import torch
from torch import nn

from morrow.carrier import LowRankStateCarrier, stable_slerp_carry
from morrow.config import ExperimentConfig
from morrow.data import load_capability_examples, load_session_sequences
from morrow.lora import WriteInterface
from morrow.trainer import MorrowModel, curriculum_depth


def test_carrier_starts_as_identity_and_has_gradients():
    carrier = LowRankStateCarrier(2, 3, 4, 5, rank=2)
    state = torch.randn(2, 3, 4, 5, requires_grad=True)
    output = carrier(state)
    torch.testing.assert_close(output, state.float())
    output.square().mean().backward()
    assert state.grad is not None
    assert carrier.key_b.grad is not None
    assert carrier.value_b.grad is not None


def test_carrier_matches_key_and_value_axis_equation():
    carrier = LowRankStateCarrier(1, 1, 2, 3, rank=1)
    state = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    with torch.no_grad():
        carrier.key_a.copy_(torch.tensor([[[1.0, 2.0]]]))
        carrier.key_b.copy_(torch.tensor([[[3.0], [4.0]]]))
        carrier.value_b.copy_(torch.tensor([[[1.0], [2.0], [3.0]]]))
        carrier.value_a.copy_(torch.tensor([[[4.0, 5.0, 6.0]]]))
        carrier.gate_identity.fill_(0.5)
        carrier.gate_key.fill_(0.25)
        carrier.gate_value.fill_(0.75)
    terminal = state[0, 0]
    key = carrier.key_b[0] @ carrier.key_a[0] @ terminal
    value = terminal @ carrier.value_b[0] @ carrier.value_a[0]
    expected = 0.5 * terminal + 0.25 * key + 0.75 * value
    torch.testing.assert_close(carrier(state)[0, 0], expected)


def test_global_slerp_preserves_previous_norm():
    previous = torch.randn(3, 2, 4, 5)
    candidate = torch.randn_like(previous)
    carried = stable_slerp_carry(previous, candidate, 0.75)
    torch.testing.assert_close(
        torch.linalg.vector_norm(carried),
        torch.linalg.vector_norm(previous),
        rtol=1e-5,
        atol=1e-5,
    )


def test_zero_previous_selects_first_candidate():
    previous = torch.zeros(2, 2)
    candidate = torch.randn(2, 2)
    torch.testing.assert_close(stable_slerp_carry(previous, candidate, 0.75), candidate)


def test_antipodal_slerp_is_fail_closed():
    previous = torch.tensor([2.0, 0.0])
    candidate = torch.tensor([-4.0, 0.0])
    torch.testing.assert_close(stable_slerp_carry(previous, candidate, 0.75), previous)


def test_curriculum_reaches_requested_depth():
    assert curriculum_depth(0, 100, 10) == 1
    assert curriculum_depth(50, 100, 10) == 5
    assert curriculum_depth(100, 100, 10) == 10
    assert curriculum_depth(1000, 100, 10) == 10


def test_write_lora_is_disabled_outside_write_context():
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    model.requires_grad_(False)
    writer = WriteInterface(model, 4, 2, 2, 2.0, 0.0, ["all-linear"])
    adapter = next(iter(writer.adapters.values()))
    nn.init.ones_(adapter.up.weight)
    hidden = torch.randn(1, 4)
    without = model(hidden)
    with writer.writing():
        during = model(hidden)
    after = model(hidden)
    assert not torch.allclose(without, during)
    torch.testing.assert_close(without, after)


def test_write_lora_accepts_mixed_precision_hidden_states():
    model = nn.Sequential(nn.Linear(4, 4, bias=False).to(torch.bfloat16))
    model.requires_grad_(False)
    writer = WriteInterface(model, 4, 2, 2, 2.0, 0.0, ["all-linear"])
    hidden = torch.randn(1, 4, dtype=torch.bfloat16)
    with writer.writing():
        output = model(hidden)
    assert output.dtype == torch.bfloat16


def test_strict_data_interfaces_and_source_ids(tmp_path):
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        json.dumps(
            {
                "sequence_id": "m1",
                "sessions": [
                    {
                        "context": "context",
                        "qa": [{"question": "question", "answer": "answer"}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    capability = tmp_path / "capability.jsonl"
    capability.write_text(
        json.dumps({"source_id": "g1", "prompt": "prompt", "completion": "completion"})
        + "\n",
        encoding="utf-8",
    )
    assert load_session_sequences(sessions)[0].sequence_id == "m1"
    assert load_capability_examples(capability)[0].source_id == "g1"


def test_configuration_rejects_invalid_tbptt():
    config = ExperimentConfig()
    config.training.max_unroll = 2
    config.training.tbptt_horizon = 3
    with pytest.raises(ValueError):
        config.validate()


def test_public_checkpoint_contains_only_writer_and_carrier(tmp_path):
    model = MorrowModel(
        backbone=nn.Identity(),
        writer=nn.Linear(3, 2),
        carrier=nn.Linear(2, 2),
        interpolation=0.75,
        epsilon=1e-6,
        max_context_tokens=16,
        max_query_tokens=16,
    )
    state = model.public_state_dict()
    assert state
    assert all(key.startswith(("writer.", "carrier.")) for key in state)
    from safetensors.torch import save_file

    path = tmp_path / "checkpoint.safetensors"
    save_file(state, path)
    with torch.no_grad():
        model.writer.weight.zero_()
    model.load_public_checkpoint(path)
    assert not torch.equal(model.writer.weight, torch.zeros_like(model.writer.weight))
