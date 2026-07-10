# Environment Notes

This repository contains scripts and compact analysis outputs. It does not package the full local execution environment.

The original runs used two separate environments:

```text
KARL environment:
  Python 3.10
  PyTorch / torchvision with CUDA
  omegaconf, einops, timm, opencv-python, tqdm
  official KARL code and VQGAN checkpoint

Qwen environment:
  Qwen2.5-VL dependencies
  qwen-vl-utils
  local Qwen2.5-VL model access
```

Required external assets:

```text
Perception Test train MCQ annotations and videos
KARL VQGAN checkpoint
karl_small VQGAN quantized-latents checkpoint
Qwen2.5-VL model
```

Large videos, checkpoints, reconstructions, and raw predictions are excluded from this repository.
