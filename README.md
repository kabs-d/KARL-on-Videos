# KARL on Videos

This repository is an exploratory extension of **KARL** from images to short videos. The project asks what KARL's adaptive image tokenizer preserves, discards, and organizes when applied frame-by-frame to video QA examples from the **Perception Test** train MCQ split.

The repository is organized around four analysis directions:

1. **KARL reconstructions and downstream VLM behavior**  
   Feed KARL-reconstructed video frames into Qwen and measure how multiple-choice QA accuracy changes across task tags and epsilon thresholds.

2. **Epsilon vs token utilization over video frames**  
   Track how active KARL token count changes with epsilon and varies over frames from the same video.

3. **Higher compression keeps more distinct tokens**  
   Test whether the smaller active-token set at higher epsilon becomes less redundant by comparing read-attention overlap among surviving latent tokens.

4. **Object-like and temporally persistent read attention**  
   Visualize KARL read attention maps and inspect whether the same latent index can keep attending to a similar object-like region as it moves across frames.

The current pushed release packages the first direction completely, with compact aggregate reports and scripts. The remaining directions were run locally during the project and are being packaged into similarly compact GitHub artifacts.

## Current Packaged Result: Qwen on KARL Reconstructions

For the first direction, we evaluate Qwen on original video frames versus KARL-reconstructed frames.

Final combined analysis set:

```text
664 MCQs
324 unique videos
8 uniformly sampled frames per video/question
```

| condition | rows | Qwen accuracy | mean active KARL tokens | reconstruction L1 |
|---|---:|---:|---:|---:|
| original frames | 664 | 0.6280 | 256.00 | 0.00000 |
| KARL eps=0.03 | 664 | 0.5858 | 248.40 | 0.04296 |
| KARL eps=0.05 | 664 | 0.5512 | 191.16 | 0.04705 |
| KARL eps=0.07 | 664 | 0.5301 | 111.10 | 0.05964 |

The strongest compression-sensitive tags at `eps=0.07` were:

| tag | original | eps=0.07 | delta |
|---|---:|---:|---:|
| part recognition | 0.7931 | 0.5172 | -0.2759 |
| place recognition | 0.9091 | 0.6591 | -0.2500 |
| object recognition | 0.7327 | 0.5446 | -0.1881 |
| object counting | 0.5909 | 0.4818 | -0.1091 |

Motion, occlusion, object permanence, and solidity were comparatively stable or slightly improved in this run.

## Dataset Construction

We do **not** introduce new question annotations. The questions, answer options, answer IDs, tags, reasoning labels, and videos come from the official Perception Test train MCQ annotations.

Our contribution is the analysis curation:

1. Parse official train MCQ annotations.
2. Verify that referenced train videos and 3-option MCQs are available locally.
3. Map official fine-grained tags into five broader visual task families.
4. Build a balanced task subset and a same-video multi-question control subset.
5. Combine both subsets and deduplicate by `row_uid = split:video_id:question_id`.

See [docs/dataset_construction.md](docs/dataset_construction.md) for details.

## Packaged Results

Experiment 1, Qwen on KARL reconstructions:

- [Experiment note](docs/experiment_1_qwen_karl_tradeoff.md)
- [Combined Qwen/KARL summary](results/combined_qwen_karl_v1/reports/combined_qwen_karl_tradeoff_summary.md)
- [Family accuracy table](results/combined_qwen_karl_v1/tables/combined_family_accuracy.csv)
- [Major-tag accuracy table](results/combined_qwen_karl_v1/tables/combined_major_tag_accuracy.csv)
- [Accuracy vs active tokens](results/combined_qwen_karl_v1/figures/combined_accuracy_vs_active_tokens.png)
- [Tag accuracy heatmap](results/combined_qwen_karl_v1/figures/combined_tag_accuracy_heatmap.png)

Same-video control support:

- [Same-video question effects](results/same_video_question_effects_v1/reports/same_video_question_effects_summary.md)
- [Same-video case index](results/same_video_question_effects_v1/tables/same_video_case_index.csv)

## Planned Packaging For Remaining Directions

The next release artifacts will add:

```text
results/temporal_token_usage_v1/
results/latent_distinctiveness_v1/
results/read_attention_object_persistence_v1/
```

These will contain compact tables, figures, and representative visual examples rather than raw reconstructions or large attention arrays.

## Reproduction Scripts

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

The scripts assume local access to:

- Perception Test train MCQ annotations and videos
- KARL VQGAN checkpoint
- Qwen2.5-VL environment

Large data artifacts are intentionally not included in this repository.

## What Is Not Included

This repository excludes:

- Perception Test videos
- raw official annotation JSONs
- model checkpoints
- KARL reconstructions
- attention `.npz` files
- full Qwen prediction JSONLs

Only compact scripts, aggregate summaries, tables, and figures are included.

## Notes

This is exploratory analysis, not a benchmark claim. The same-video subset is intentionally enriched for clips with multiple question types, so the combined Qwen/KARL result should be read as a controlled research probe rather than a random sample of the full Perception Test train split.
