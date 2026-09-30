### 1 x NVIDIA L40S

AWS g6e.2xlarge (8 vCPU, 1 x L40S 48 GB) on Anyscale. Job config: `anyscale/l40s.yaml`.

8 images, 1024px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 108.3 | 4.43 | 1.0x | 13.1 |
| `cuda:2@0.5` | 2 | 0.5 | 1 | 141.3 | 3.4 | 0.8x | 33.6 |
