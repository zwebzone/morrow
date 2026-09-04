# Data interface

The release intentionally contains no benchmark examples. Convert licensed or
public data into the two JSONL interfaces below.

## Session sequences

Each line represents one chronologically ordered sequence. `sequence_id` must
identify the original source (for example, a conversation), rather than an
individual QA pair.

```json
{"sequence_id":"memory-source-0001","sessions":[{"context":"Session-one text.","qa":[{"question":"Question about session one?","answer":"Answer one."}]},{"context":"Session-two text.","qa":[{"question":"Question about session two?","answer":"Answer two."}]}]}
```

## Capability preservation

Each line contains a prompt and its gold continuation. These examples must be
unrelated to the memory sequences. The loader rejects an exact overlap between
capability `source_id` values and memory `sequence_id` values.

```json
{"source_id":"general-source-0001","prompt":"Solve: 7 + 5 =","completion":"12"}
```

Data preparation must keep validation and test sources separate from this
training file. No benchmark-specific preprocessing or answer normalization is
implemented in the training package.

