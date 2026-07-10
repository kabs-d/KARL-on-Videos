# Dataset Construction

This project uses the official **Perception Test train MCQ** annotations. The official data already provides each multiple-choice question with:

```text
video_id
question
3 answer options
answer_id
area
reasoning
tag
```

The official rows are parsed into `manifests/train_mcq.jsonl` by `scripts/build_mcq_manifest.py`.

## Official Files Used

```text
data/annotations/train/mcq/mc_question_train.json
data/challenge_ids/mc_question_train_id_list.csv
data/cut_mappings/cut_frame_mapping_train.json
data/videos/train/videos/*.mp4
```

The manifest builder restricts rows to the official train MCQ challenge video IDs and verifies that the corresponding local video file and 3-option MCQ format exist.

In this setup:

```text
official train MCQ rows: 7392
valid local MCQ rows:   7392
skipped missing video:  0
skipped bad options:   0
```

## Task-Family Curation

The official Perception Test tags are fine-grained. For analysis, we map them into five broader visual task families:

| family | official tags used |
|---|---|
| occlusion_permanence | occlusion, object permanence, containment, solidity, collision |
| object_counting | object counting |
| spatial_relations | spatial relations, part recognition |
| temporal_event | motion, sequencing, task completion, action counting, event counting, event recall |
| recognition_control | object recognition, place recognition, action recognition |

Rows whose tags do not match these concrete visual families are excluded from this analysis.

```text
official train MCQ rows:       7392
concrete curated MCQ rows:     6507
curated unique videos:         1934
```

## Analysis Subsets

Two curated subsets are used.

### Main Balanced Subset

`main_balanced_300_seed17`

```text
300 rows
285 unique videos
60 rows per primary task family
```

This subset gives a broad task-balanced view.

### Same-Video Control Subset

`same_video_controls_seed17`

```text
385 rows
60 unique videos
5-11 questions per video
```

This subset intentionally selects videos that have multiple official questions spanning several task families. It supports comparisons where visual evidence is fixed but question type changes.

## Combined Analysis Set

The primary reported result combines the two subsets and deduplicates exact overlaps:

```text
main balanced rows:       300
same-video rows:          385
exact row_uid overlaps:   21
combined rows:            664
combined unique videos:   324
```

Deduplication key:

```text
row_uid = split:video_id:question_id
```

For exact overlaps, the same-video run is preferred as the prediction source so that same-video case-study tables and the combined aggregate remain consistent.

## Important Caveat

This is not a new dataset and does not introduce new ground-truth labels. It is a curated analysis subset of Perception Test train MCQ, designed to probe KARL reconstruction behavior and task sensitivity.
