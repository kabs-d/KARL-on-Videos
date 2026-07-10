# Direction 1: Object-Like And Temporally Persistent Read Attention

This direction asks whether individual KARL latent indices show visually interpretable read behavior when KARL is applied frame-by-frame to short videos.

The analysis is intentionally manual and narrow. It does not use YOLO, segmentation labels, or object ground truth. The goal is to document whether selected active latent indices produce compact read-attention maps that visually align with salient regions such as cups, hands, table edges, or clothing.

## What Is Being Visualized

The map shown here is KARL's read attention:

```text
encoder latent token query -> original 16x16 input grid key
```

For each active latent index, this gives a `16 x 16` attention map over the input image grid. I upsample each map to `256 x 256` only for visualization. The color scale is contrast-normalized independently per map, so the images should be read as "where this latent index concentrates most in this frame," not as absolute attention magnitudes across different tokens.

The stable identifier in this note is the **latent index**. Some older generated filenames used both `token_rank` and `latent`, such as `token_rank_159_latent_159.png`; the analysis here tracks `latent_159`. In these selected examples, active rank and latent index happen to match, but latent index is the identity used for frame-to-frame comparison.

## Manual Protocol

I inspected token-wise read-attention maps for three cup-moving videos at `eps=0.07`. This note packages representative examples from `video_76` and `video_1614`, because their first-frame maps were visually clean and easy to audit.

Manual descriptions below are visual annotations, not ground-truth labels. They should be treated as qualitative evidence that some active KARL latent indices can concentrate on object-like or region-like image areas.

## Video Preview

`video_76` sampled frames:

<video src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames.mp4" controls width="480"></video>

[Open the sampled-frame MP4](../results/direction1_object_read_attention_v1/media/video_76_sampled_frames.mp4)

## First-Frame Maps: `video_76`

Original first sampled frame:

<img src="../results/direction1_object_read_attention_v1/original_frames/video_76_frame_000.png" width="256">

| latent index | attention heatmap | manual visual description |
|---:|---|---|
| 36 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_036.png" width="112"> | compact attention on a cup on the table |
| 38 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_038.png" width="112"> | compact attention on a cup on the table |
| 39 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_039.png" width="112"> | compact attention on a cup on the table |
| 42 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_042.png" width="112"> | compact attention on a cup on the table |
| 159 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_159.png" width="112"> | compact attention on a cup on the table |
| 132 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_132.png" width="112"> | attention concentrated near the hand |
| 158 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_158.png" width="112"> | attention spread over the three-cup group |

## First-Frame Maps: `video_1614`

Original first sampled frame:

<img src="../results/direction1_object_read_attention_v1/original_frames/video_1614_frame_000.png" width="256">

| latent index | attention heatmap | manual visual description |
|---:|---|---|
| 0 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_000.png" width="112"> | cup |
| 3 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_003.png" width="112"> | table leg |
| 4 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_004.png" width="112"> | cup |
| 17 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_017.png" width="112"> | cup |
| 20 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_020.png" width="112"> | cup |
| 25 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_025.png" width="112"> | table edge |
| 35 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_035.png" width="112"> | cup |
| 53 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_053.png" width="112"> | cup |
| 103 | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_103.png" width="112"> | cup |

## Temporal Concentration In `video_76`

The next table tracks selected latent indices across the eight sampled frames of `video_76`. I do **not** claim object tracking here. The safer observation is that some latent indices remain spatially concentrated across frames instead of diffusing broadly. In several cases, the concentrated region stays in a similar part of the frame.

| latent index | visual note | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 |
|---:|---|---|---|---|---|---|---|---|---|
| 36 | left-cup spatial area | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_000.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_001.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_002.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_003.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_004.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_005.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_006.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_007.png" width="64"> |
| 39 | cup-like spatial area | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_000.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_001.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_002.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_003.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_004.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_005.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_006.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_007.png" width="64"> |
| 42 | cup-like spatial area | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_000.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_001.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_002.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_003.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_004.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_005.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_006.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_007.png" width="64"> |
| 43 | shirt/t-shirt spatial area | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_000.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_001.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_002.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_003.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_004.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_005.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_006.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_007.png" width="64"> |
| 159 | cup-like spatial area when active | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_000.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_001.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_002.png" width="64"> | inactive | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_004.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_005.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_006.png" width="64"> | <img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_007.png" width="64"> |

## Takeaways

These examples suggest that KARL read attention can be visually interpretable at the latent-index level. Some selected active latent indices produce compact maps that line up with object-like regions, and several remain spatially concentrated over sampled frames from the same video.

The evidence is qualitative, but it is useful because it looks inside the tokenizer itself rather than only measuring downstream QA performance. A stronger follow-up would scale this beyond manual inspection by pairing read maps with verified object masks or carefully audited human labels.

## Guardrails

- These are attention maps, not causal attribution maps.
- The labels in the tables are manual visual descriptions, not official annotations.
- The temporal table shows spatial concentration and persistence, not proven object following.
- Since no segmentation ground truth is used, this direction should be read as an interpretability probe rather than an object-localization benchmark.

## Artifacts

- Selected attention heatmaps: [results/direction1_object_read_attention_v1/attention_heatmaps](../results/direction1_object_read_attention_v1/attention_heatmaps)
- Sampled-frame MP4: [results/direction1_object_read_attention_v1/media/video_76_sampled_frames.mp4](../results/direction1_object_read_attention_v1/media/video_76_sampled_frames.mp4)
- Asset manifest: [results/direction1_object_read_attention_v1/tables/selected_read_attention_assets.csv](../results/direction1_object_read_attention_v1/tables/selected_read_attention_assets.csv)
- Rendering script: [scripts/render_direction1_read_attention_assets.py](../scripts/render_direction1_read_attention_assets.py)
