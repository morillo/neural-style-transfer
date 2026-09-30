### 1 x NVIDIA L40S

AWS g6e.2xlarge (8 vCPU, 1 x L40S 48 GB) on Anyscale. Job config: `anyscale/l40s.yaml`.

8 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 41.9 | 11.46 | 1.0x | 4.4 |
| `cuda:2@0.5` | 2 | 0.5 | 1 | 60.8 | 7.9 | 0.7x | 13.3 |
| `cuda:4@0.25` | 4 | 0.25 | 1 | 58.5 | 8.2 | 0.7x | 21.9 |
