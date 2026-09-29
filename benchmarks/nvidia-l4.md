### 1 x NVIDIA L4

AWS g6.2xlarge (8 vCPU, 1 x L4 24 GB) on Anyscale. Job config: `anyscale/l4.yaml`.

8 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 79.3 | 6.06 | 1.0x | 9.3 |
| `cuda:2@0.5` | 2 | 0.5 | 1 | 107.5 | 4.46 | 0.7x | 24.8 |
| `cuda:4@0.25` | 4 | 0.25 | 1 | 108.9 | 4.41 | 0.7x | 50.7 |
