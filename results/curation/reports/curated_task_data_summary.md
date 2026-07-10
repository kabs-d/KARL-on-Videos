# Curated Perception Test Task Data

Scope: train MCQ rows assigned to concrete KARL-QA task families.

## Overview

- source train rows: 7392
- curated rows: 6507
- excluded other rows: 885
- curated unique videos: 1934
- same-video candidate videos: 1419
- calibration/main overlap rows: 0

## Primary Family Counts

- object_counting: 1132
- occlusion_permanence: 637
- recognition_control: 1254
- spatial_relations: 1584
- temporal_event: 1900

## Multi-Label Membership Counts

- object_counting: 1278
- occlusion_permanence: 637
- recognition_control: 2179
- spatial_relations: 1926
- temporal_event: 3151

## Subsets

### calibration_balanced_150_seed17

- rows: 150
- unique videos: 138
- primary family:
  - object_counting: 30
  - occlusion_permanence: 30
  - recognition_control: 30
  - spatial_relations: 30
  - temporal_event: 30

### main_balanced_300_seed17

- rows: 300
- unique videos: 285
- primary family:
  - object_counting: 60
  - occlusion_permanence: 60
  - recognition_control: 60
  - spatial_relations: 60
  - temporal_event: 60

### same_video_controls_seed17

- rows: 385
- unique videos: 60
- primary family:
  - object_counting: 40
  - occlusion_permanence: 75
  - recognition_control: 62
  - spatial_relations: 128
  - temporal_event: 80

### temporal_signature_bank_seed17

- rows: 60
- unique videos: 60
- primary family:
  - occlusion_permanence: 20
  - recognition_control: 20
  - temporal_event: 20

### token_responsibility_bank_seed17

- rows: 40
- unique videos: 39
- primary family:
  - object_counting: 8
  - occlusion_permanence: 8
  - recognition_control: 8
  - spatial_relations: 8
  - temporal_event: 8

### task_critical_ablation_bank_seed17

- rows: 40
- unique videos: 40
- primary family:
  - object_counting: 10
  - occlusion_permanence: 10
  - recognition_control: 10
  - spatial_relations: 10
