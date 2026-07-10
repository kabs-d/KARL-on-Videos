# KARL on Videos

This repository is an exploratory extension of **KARL** from images to short videos. The central question is:

> What does KARL's adaptive image tokenizer preserve, discard, and organize when it is applied frame-by-frame to video QA examples?

The project uses short videos from the **Perception Test** train MCQ split and treats KARL as an image tokenizer applied to sampled video frames. The goal is not to claim a new benchmark result, but to build focused analysis probes around token usage, compression, reconstruction, and latent attention behavior.

## Analysis Directions

The repository is organized as four linked analysis notes. I am packaging them one by one so each can be read independently.

| direction | status | note |
|---|---|---|
| 1. KARL reconstructions and downstream VLM behavior | packaged | [analysis README](docs/experiment_1_qwen_karl_tradeoff.md) |
| 2. Epsilon vs token utilization over video frames | next | README coming next |
| 3. Higher compression keeps more distinct tokens | next | README coming next |
| 4. Object-like and temporally persistent read attention | next | README coming next |

## Direction 1 Snapshot

The first packaged direction asks whether Qwen can still answer Perception Test MCQs when original frames are replaced by KARL reconstructions at different epsilon thresholds.

Combined analysis set:

```text
664 MCQs
324 unique videos
8 uniformly sampled frames per question/video
```

Global result:

| condition | Qwen accuracy | mean active KARL tokens |
|---|---:|---:|
| original frames | 0.6280 | 256.00 |
| KARL eps=0.03 | 0.5858 | 248.40 |
| KARL eps=0.05 | 0.5512 | 191.16 |
| KARL eps=0.07 | 0.5301 | 111.10 |

The main pattern is task-dependent compression sensitivity: recognition/detail-heavy tags degrade most, while motion and occlusion-style tags are more stable in this run. The same-video control subset also shows that different questions on the same compressed visual evidence can be helped or hurt differently.

See the detailed note: [Direction 1: KARL Reconstructions and Downstream VLM Behavior](docs/experiment_1_qwen_karl_tradeoff.md).

## Dataset Construction

No new question annotations are introduced. The questions, answer options, answer IDs, tags, reasoning labels, and videos come from the official Perception Test train MCQ annotations.

The curation steps are:

1. Parse official train MCQ annotations.
2. Verify that referenced train videos and 3-option MCQs are available locally.
3. Map official fine-grained tags into five broader visual task families.
4. Build a balanced task subset and a same-video multi-question control subset.
5. Combine both subsets and deduplicate exact overlaps by `row_uid = split:video_id:question_id`.

See [docs/dataset_construction.md](docs/dataset_construction.md) for the exact construction.

## Included Artifacts

This repository includes compact scripts, aggregate reports, tables, and figures. It intentionally excludes large data artifacts.

Included for Direction 1:

- [Combined Qwen/KARL summary](results/combined_qwen_karl_v1/reports/combined_qwen_karl_tradeoff_summary.md)
- [Major-tag accuracy table](results/combined_qwen_karl_v1/tables/combined_major_tag_accuracy.csv)
- [Family accuracy table](results/combined_qwen_karl_v1/tables/combined_family_accuracy.csv)
- [Accuracy vs active tokens](results/combined_qwen_karl_v1/figures/combined_accuracy_vs_active_tokens.png)
- [Tag accuracy heatmap](results/combined_qwen_karl_v1/figures/combined_tag_accuracy_heatmap.png)
- [Same-video question effects](results/same_video_question_effects_v1/reports/same_video_question_effects_summary.md)

Excluded:

- Perception Test videos
- raw official annotation JSONs
- model checkpoints
- KARL reconstructions
- attention `.npz` files
- full Qwen prediction JSONLs

## Scripts

Scripts are under [scripts/](scripts):

```text
build_mcq_manifest.py
curate_task_data.py
run_qwen_perception_calibration.py
run_karl_reconstruction_mdl.py
run_qwen_on_karl_reconstructions.py
analyze_combined_qwen_karl_tradeoff.py
analyze_same_video_question_effects.py
```

They assume local access to the Perception Test train MCQ data, a KARL VQGAN checkpoint, and a Qwen2.5-VL environment.
