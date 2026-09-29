### 1 x NVIDIA A10G

AWS g5.2xlarge (8 vCPU, 1 x A10G 24 GB) on Anyscale. Job config: `anyscale/a10g.yaml`.

8 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 85.7 | 5.6 | 1.0x | 9.8 |
| `cuda:2@0.5` | 2 | 0.5 | 1 | 116.8 | 4.11 | 0.7x | 27.0 |
| `cuda:4@0.25` | 4 | 0.25 | 1 | 116.4 | 4.12 | 0.7x | 51.2 |
