### Apple M4 Max

8 images, 512px longest side, 300 L-BFGS steps each. torch 2.8.0, ray 2.49.1. Wall clock includes starting the Ray actors and loading VGG19.

| Config | Workers | Share of accelerator per worker | Wall clock (s) | Images/min | Speed-up | Median s/image (per worker) |
|---|---:|---:|---:|---:|---:|---:|
| `cpu:1` | 1 | - | 478.9 | 1.0 | 1.0x | 61.7 |
| `cpu:2` | 2 | - | 352.8 | 1.36 | 1.4x | 90.9 |
| `cpu:4` | 4 | - | 300.2 | 1.6 | 1.6x | 146.0 |
| `mps:1` | 1 | 1.0 | 132.5 | 3.62 | 3.6x | 16.6 |
| `mps:2@0.5` | 2 | 0.5 | 93.9 | 5.11 | 5.1x | 23.0 |
| `mps:4@0.25` | 4 | 0.25 | 84.8 | 5.66 | 5.7x | 40.7 |
