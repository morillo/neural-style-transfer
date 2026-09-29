### 4 x NVIDIA A10G (one node)

AWS g5.12xlarge (48 vCPU, 4 x A10G 24 GB) on Anyscale. Job config: `anyscale/4xa10g.yaml`.

32 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 319.9 | 6.0 | 1.0x | 9.7 |
| `cuda:2` | 2 | 1.0 | 1 | 164.7 | 11.66 | 1.9x | 9.7 |
| `cuda:4` | 4 | 1.0 | 1 | 96.3 | 19.94 | 3.3x | 9.7 |
