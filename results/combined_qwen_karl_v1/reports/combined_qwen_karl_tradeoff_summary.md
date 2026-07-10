# Combined Qwen/KARL Reconstruction Tradeoff

This is the primary overall analysis for the Qwen side of the KARL extension.

## Dataset Construction

The dataset is derived from the Perception Test train MCQ manifest.

- source train MCQ rows: 7392
- rows with valid local videos and 3 MCQ options: 7392
- concrete curated rows after task-family assignment: 6507
- curated unique videos: 1934
- main balanced subset: 300 rows, 285 videos
- same-video control subset: 385 rows, 60 videos
- exact row overlap between subsets: 21
- combined deduplicated analysis set: 664 rows, 324 videos

Task families were assigned from Perception Test tags using the curation script, then the combined analysis deduplicated exact `row_uid` overlaps.

## Global Tradeoff

| condition | rows | accuracy | delta | active mean | L1 mean | fixed | lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 664 | 0.6280 | 0.0000 | 256.00 | 0.00000 | 0 | 0 |
| eps_003 | 664 | 0.5858 | -0.0422 | 248.40 | 0.04296 | 38 | 66 |
| eps_005 | 664 | 0.5512 | -0.0768 | 191.16 | 0.04705 | 41 | 92 |
| eps_007 | 664 | 0.5301 | -0.0979 | 111.10 | 0.05964 | 56 | 121 |

## By Major Tag

| tag | n | original | eps003 | eps005 | eps007 | eps007 delta |
| --- | --- | --- | --- | --- | --- | --- |
| spatial relations | 210 | 0.5238 | 0.5000 | 0.4429 | 0.4476 | -0.0762 |
| motion | 172 | 0.5233 | 0.5407 | 0.5233 | 0.5814 | 0.0581 |
| object recognition | 101 | 0.7327 | 0.5842 | 0.5446 | 0.5446 | -0.1881 |
| place recognition | 88 | 0.9091 | 0.8182 | 0.8182 | 0.6591 | -0.2500 |
| solidity | 68 | 0.3971 | 0.3971 | 0.3676 | 0.4265 | 0.0294 |
| sequencing | 80 | 0.6625 | 0.6250 | 0.6000 | 0.6125 | -0.0500 |
| occlusion | 65 | 0.3846 | 0.4308 | 0.3846 | 0.4462 | 0.0615 |
| object permanence | 65 | 0.3846 | 0.4308 | 0.3846 | 0.4462 | 0.0615 |
| object counting | 110 | 0.5909 | 0.5455 | 0.5000 | 0.4818 | -0.1091 |
| collision | 30 | 0.7667 | 0.6333 | 0.7000 | 0.6333 | -0.1333 |
| action counting | 45 | 0.3556 | 0.2889 | 0.2889 | 0.2667 | -0.0889 |
| part recognition | 29 | 0.7931 | 0.7586 | 0.6552 | 0.5172 | -0.2759 |

## Most Compression-Sensitive Tags At eps007

Only tags with negative accuracy delta are included here; stable or improved tags remain in the full major-tag table above.

| tag | n | original | eps007 | delta |
| --- | --- | --- | --- | --- |
| part recognition | 29 | 0.7931 | 0.5172 | -0.2759 |
| place recognition | 88 | 0.9091 | 0.6591 | -0.2500 |
| object recognition | 101 | 0.7327 | 0.5446 | -0.1881 |
| collision | 30 | 0.7667 | 0.6333 | -0.1333 |
| object counting | 110 | 0.5909 | 0.4818 | -0.1091 |
| action counting | 45 | 0.3556 | 0.2667 | -0.0889 |
| spatial relations | 210 | 0.5238 | 0.4476 | -0.0762 |
| sequencing | 80 | 0.6625 | 0.6125 | -0.0500 |

## Interpretation

- This combined table is the headline Qwen/KARL result.
- The same-video subset is intentionally overrepresented in this union because it provides controlled multi-question videos.
- Exact overlapping questions are counted once.
- The combined report should be used for overall accuracy and tag trends.
