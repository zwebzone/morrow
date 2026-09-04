from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class QA:
    question: str
    answer: str


@dataclass(frozen=True)
class Session:
    context: str
    qa: tuple[QA, ...]


@dataclass(frozen=True)
class SessionSequence:
    sequence_id: str
    sessions: tuple[Session, ...]


@dataclass(frozen=True)
class CapabilityExample:
    source_id: str
    prompt: str
    completion: str


def _nonempty_string(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return value


def load_session_sequences(path: str | Path) -> list[SessionSequence]:
    sequences: list[SessionSequence] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sequence_id = _nonempty_string(row.get("sequence_id"), "sequence_id", line_number)
            if sequence_id in seen:
                raise ValueError(f"line {line_number}: duplicate sequence_id {sequence_id!r}")
            raw_sessions = row.get("sessions")
            if not isinstance(raw_sessions, list) or not raw_sessions:
                raise ValueError(f"line {line_number}: sessions must be a non-empty list")
            sessions: list[Session] = []
            for session_index, raw_session in enumerate(raw_sessions):
                context = _nonempty_string(
                    raw_session.get("context"), f"sessions[{session_index}].context", line_number
                )
                raw_qa = raw_session.get("qa")
                if not isinstance(raw_qa, list) or not raw_qa:
                    raise ValueError(
                        f"line {line_number}: sessions[{session_index}].qa must be non-empty"
                    )
                qa = tuple(
                    QA(
                        question=_nonempty_string(
                            item.get("question"),
                            f"sessions[{session_index}].qa[{qa_index}].question",
                            line_number,
                        ),
                        answer=_nonempty_string(
                            item.get("answer"),
                            f"sessions[{session_index}].qa[{qa_index}].answer",
                            line_number,
                        ),
                    )
                    for qa_index, item in enumerate(raw_qa)
                )
                sessions.append(Session(context=context, qa=qa))
            seen.add(sequence_id)
            sequences.append(SessionSequence(sequence_id=sequence_id, sessions=tuple(sessions)))
    if not sequences:
        raise ValueError(f"no session sequences found in {path}")
    return sequences


def load_capability_examples(path: str | Path) -> list[CapabilityExample]:
    examples: list[CapabilityExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                CapabilityExample(
                    source_id=_nonempty_string(row.get("source_id"), "source_id", line_number),
                    prompt=_nonempty_string(row.get("prompt"), "prompt", line_number),
                    completion=_nonempty_string(row.get("completion"), "completion", line_number),
                )
            )
    if not examples:
        raise ValueError(f"no capability examples found in {path}")
    return examples


class SequenceSampler:
    def __init__(self, sequences: Iterable[SessionSequence], seed: int):
        self.sequences = tuple(sequences)
        if not self.sequences:
            raise ValueError("sequences cannot be empty")
        self.random = random.Random(seed)

    def sample(self, maximum_depth: int) -> tuple[SessionSequence, int]:
        sequence = self.random.choice(self.sequences)
        depth = self.random.randint(1, min(maximum_depth, len(sequence.sessions)))
        return sequence, depth

    def sample_items(self, items: tuple[QA, ...] | list[QA], count: int) -> list[QA]:
        if count <= 0 or not items:
            return []
        return [self.random.choice(items) for _ in range(count)]
