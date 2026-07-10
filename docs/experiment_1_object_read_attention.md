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

The original frame is shown in the first column, followed by selected latent-index read maps.

<table>
  <tr>
    <th>original</th>
    <th>latent 36</th>
    <th>latent 38</th>
    <th>latent 39</th>
    <th>latent 42</th>
    <th>latent 159</th>
    <th>latent 132</th>
    <th>latent 158</th>
  </tr>
  <tr>
    <td><img src="../results/direction1_object_read_attention_v1/original_frames/video_76_frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_036.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_038.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_039.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_042.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_159.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_132.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_76/latent_158.png" width="128"></td>
  </tr>
  <tr>
    <td>source frame</td>
    <td>cup</td>
    <td>cup</td>
    <td>cup</td>
    <td>cup</td>
    <td>cup</td>
    <td>hand</td>
    <td>three-cup group</td>
  </tr>
</table>

## First-Frame Maps: `video_1614`

The original frame is again kept in the first column for direct side-by-side comparison.

<table>
  <tr>
    <th>original</th>
    <th>latent 0</th>
    <th>latent 3</th>
    <th>latent 4</th>
    <th>latent 17</th>
    <th>latent 20</th>
    <th>latent 25</th>
    <th>latent 35</th>
    <th>latent 53</th>
    <th>latent 103</th>
  </tr>
  <tr>
    <td><img src="../results/direction1_object_read_attention_v1/original_frames/video_1614_frame_000.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_000.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_003.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_004.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_017.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_020.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_025.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_035.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_053.png" width="112"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/first_frame/video_1614/latent_103.png" width="112"></td>
  </tr>
  <tr>
    <td>source frame</td>
    <td>cup</td>
    <td>table leg</td>
    <td>cup</td>
    <td>cup</td>
    <td>cup</td>
    <td>table edge</td>
    <td>cup</td>
    <td>cup</td>
    <td>cup</td>
  </tr>
</table>

## Temporal Concentration In `video_76`

The next table tracks selected latent indices across the eight sampled frames of `video_76`. I do **not** claim object tracking here. The safer observation is that some latent indices remain spatially concentrated across frames instead of diffusing broadly. In several cases, the concentrated region stays in a similar part of the frame.

<table>
  <tr>
    <th>latent index</th>
    <th>visual note</th>
    <th>f0</th>
    <th>f1</th>
    <th>f2</th>
    <th>f3</th>
    <th>f4</th>
    <th>f5</th>
    <th>f6</th>
    <th>f7</th>
  </tr>
  <tr>
    <td>source</td>
    <td>original color frame</td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_002.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_003.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames/frame_007.png" width="128"></td>
  </tr>
  <tr>
    <td>36</td>
    <td>left-cup spatial area</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_002.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_003.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_036/frame_007.png" width="128"></td>
  </tr>
  <tr>
    <td>39</td>
    <td>cup-like spatial area</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_002.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_003.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_039/frame_007.png" width="128"></td>
  </tr>
  <tr>
    <td>42</td>
    <td>cup-like spatial area</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_002.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_003.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_042/frame_007.png" width="128"></td>
  </tr>
  <tr>
    <td>43</td>
    <td>shirt/t-shirt spatial area</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_002.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_003.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_043/frame_007.png" width="128"></td>
  </tr>
  <tr>
    <td>159</td>
    <td>cup-like spatial area when active</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_000.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_001.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_002.png" width="128"></td>
    <td>inactive</td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_004.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_005.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_006.png" width="128"></td>
    <td><img src="../results/direction1_object_read_attention_v1/attention_heatmaps/temporal/video_76/latent_159/frame_007.png" width="128"></td>
  </tr>
</table>

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
