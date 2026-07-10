# Experiment 1: Qwen on KARL Reconstructions

## Question

When original video frames are replaced by KARL reconstructions, how much does Qwen's multiple-choice video QA accuracy change across task types and reconstruction-quality thresholds?

## Setup

- Dataset: combined curated Perception Test train MCQ subset
- Questions: 664 deduplicated MCQs
- Videos: 324 unique videos
- Frame sampling: 8 uniformly sampled frames per question/video
- VLM: Qwen2.5-VL
- KARL path: `karl_small` with VQGAN quantized latents
- Epsilon thresholds: `0.03`, `0.05`, `0.07`

## Conditions

| condition | frame source |
|---|---|
| original | original sampled RGB frames |
| eps=0.03 | KARL reconstructions with epsilon 0.03 |
| eps=0.05 | KARL reconstructions with epsilon 0.05 |
| eps=0.07 | KARL reconstructions with epsilon 0.07 |

## Headline Result

| condition | Qwen accuracy | mean active tokens |
|---|---:|---:|
| original | 0.6280 | 256.00 |
| eps=0.03 | 0.5858 | 248.40 |
| eps=0.05 | 0.5512 | 191.16 |
| eps=0.07 | 0.5301 | 111.10 |

Accuracy drops as compression becomes stronger, but the degradation is not uniform across tasks.

## Task Sensitivity

At `eps=0.07`, recognition-like tasks are most compression-sensitive:

| tag | delta from original |
|---|---:|
| part recognition | -0.2759 |
| place recognition | -0.2500 |
| object recognition | -0.1881 |

Temporal/occlusion-oriented tags are more stable in this run:

| tag | delta from original |
|---|---:|
| motion | +0.0581 |
| occlusion | +0.0615 |
| object permanence | +0.0615 |

## Same-Video Control

The same-video subset lets us compare multiple questions about the same video. At `eps=0.07`:

```text
50 / 60 videos had at least one changed question outcome
26 / 60 had at least one fixed question
44 / 60 had at least one lost question
20 / 60 had both fixed and lost questions on the same video
```

This supports the interpretation that compression sensitivity depends on the question's required evidence, not only the underlying video.

## Interpretation

KARL reconstructions preserve enough information for many MCQ decisions even with a much smaller active token budget, but compression affects question types differently. Recognition and fine part/detail questions degrade more sharply, while some motion and occlusion-style questions are comparatively robust.
