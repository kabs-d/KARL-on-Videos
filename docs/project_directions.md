# Project Directions

This project is structured as four complementary probes of KARL on short videos.

## 1. Object-Like And Temporally Persistent Attention Maps

Question:

> Do individual KARL latent indices attend to semantically coherent regions, and can the same latent index remain spatially concentrated across sampled frames?

Status:

```text
packaged in this repository
```

Packaged artifacts:

```text
docs/experiment_1_object_read_attention.md
results/direction1_object_read_attention_v1/
```

Interpretation target:

These examples do not prove object semantics in a causal sense. They provide visual evidence that selected active latent indices can produce compact, object-like attention maps and can remain spatially concentrated across sampled video frames.

## 2. KARL Reconstructions And Downstream VLM Behavior

Question:

> If Qwen receives KARL-reconstructed frames instead of original frames, which question types remain answerable and which degrade?

Status:

```text
packaged in this repository
```

Packaged artifacts:

```text
docs/experiment_2_qwen_karl_tradeoff.md
results/combined_qwen_karl_v1/
```

Main signal:

```text
original accuracy: 0.6280
eps=0.03:          0.5858
eps=0.05:          0.5512
eps=0.07:          0.5301
```

Recognition/detail-heavy tags degrade most, while motion and occlusion-style tags are more stable in this run.

## 3. Higher Compression Keeps More Distinct Tokens

Question:

> When epsilon increases and fewer tokens remain active, do the remaining tokens collapse onto the same regions, or do they become more distinct?

Local analysis signal:

```text
eps=0.03: ~251 active tokens, attention distinctness 0.54
eps=0.05: ~199 active tokens, attention distinctness 0.79
eps=0.07: ~116 active tokens, attention distinctness 0.86
```

Status:

```text
packaged in this repository
```

Packaged artifacts:

```text
docs/experiment_3_latent_distinctiveness.md
results/latent_distinctiveness_v1/
```

Interpretation:

Higher compression keeps fewer active latent tokens, but those surviving tokens have less-overlapping attention maps. This suggests KARL concentrates its limited token budget over more distinct visual regions.

## 4. Epsilon vs Token Utilization Over Video Frames

Question:

> How does KARL's active token count vary across frames in the same video as epsilon changes?

Intended artifacts:

```text
frame-level active-token traces
per-video token-usage plots
summary tables by epsilon
```

Interpretation target:

KARL should not be summarized only by a single average token count. In videos, active-token usage can vary frame-to-frame, and that variation is part of the tokenizer behavior.
