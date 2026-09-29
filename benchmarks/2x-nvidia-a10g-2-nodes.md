### 2 x NVIDIA A10G (two nodes)

2 x AWS g5.2xlarge (1 x A10G each) on Anyscale. Job config: `anyscale/2node-a10g.yaml`.

16 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0+cu128, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Nodes used | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda:1` | 1 | 1.0 | 1 | 165.5 | 5.8 | 1.0x | 9.8 |
| `cuda:2` | 2 | 1.0 | 2 | 101.1 | 9.49 | 1.6x | 9.8 |
