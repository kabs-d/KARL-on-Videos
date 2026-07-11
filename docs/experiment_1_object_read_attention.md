# Direction 1: KARL Read Attention Maps

This note visualizes where selected active KARL latent indices read from in short video frames.

## Map Definition

KARL reads a frame through an encoder attention block that mixes latent-token queries with the original `16×16` image/VQGAN grid. For an active latent index `k`, I extract the encoder attention from latent query `k` to the image-grid keys:

```text
read_map(k) = mean_heads Attention(q_latent[k], K_input_grid) ∈ R^{16×16}
```

This is a read-side map: it shows which input grid locations a latent token attends to while forming its representation. It is not a decoder map, segmentation mask, or causal attribution map.

Settings used here:

- Encoder layer: `7`
- Compression threshold: `eps=0.07`
- Active latent condition: `halt_probability <= 0.75`
- Visualization: `16×16` map upsampled to `256×256`
- Color scale: contrast-normalized independently per latent/frame

`latent index` is the identity tracked across frames. Text labels are manual visual notes, not ground truth.

## Video Preview

`video_76` sampled frames:

<video controls width="640">
  <source src="../results/direction1_object_read_attention_v1/media/video_76_sampled_frames.mp4" type="video/mp4">
</video>

## First-Frame Maps: `video_76`

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

Columns are sampled frames. The first row shows the original frame; lower rows show the same latent index across time. This is a spatial-concentration view, not a causal object-tracking claim.

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

## Reading Notes

- Bright regions are high relative attention within that latent map.
- Compare each temporal column vertically against the source frame.
- `inactive` means that latent index was not active for that frame.
- These are attention visualizations, not causal attribution or segmentation results.

## Artifacts

- Selected attention heatmaps: [results/direction1_object_read_attention_v1/attention_heatmaps](../results/direction1_object_read_attention_v1/attention_heatmaps)
