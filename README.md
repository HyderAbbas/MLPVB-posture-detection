# MLPVB: Multi-Layer Perceptron Vision-Based Hybrid Model for Posture-Based Heart Attack Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

Official implementation of **"MLPVB: A Multi-Layer Perceptron Vision-Based Hybrid Technique for Posture-Based Heart Attack Detection"**.

> Heart attack posture can sometimes lead to problematic situations for patients. This work introduces the **MLPVB** hybrid model, which fuses a **MobileNet-V2** patch-embedding Vision Transformer branch with a **VGG-19** feature branch, fine-tuned using **Ada2Max** (an enhanced Adam variant), to improve posture-based heart attack detection accuracy while keeping the model lightweight.

If you use this code, please [cite the paper](#citation).

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview
- **Backbone branches:** MobileNet-V2 (patch embedding → Transformer encoder) + VGG-19 (frozen feature extractor)
- **Optimizer:** Ada2Max — a tuned variant of Adam
- **Task:** Binary classification (posture indicative of heart-attack risk vs. normal)
- **Framework:** TensorFlow / Keras

## Repository Structure
```
.
├── mlpvb_fixed.py        # Main model definition + training script
├── requirements.txt      # Python dependencies
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
The model expects an ImageDataGenerator-compatible directory layout:

```
datasetval/
├── train/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

Update the `train_dir` and `test_dir` paths at the top of `mlpvb_fixed.py` (or pass them as arguments if you refactor to argparse — see [Contributing](#contact)).

> **Note:** Add a short description here of your dataset's source, size, and any license/consent restrictions, and cite it if it is third-party.

## Usage

Train the model:

```bash
python mlpvb_fixed.py
```

This will:
1. Build the VGG-19 branch and the MobileNet-V2 + Vision Transformer branch.
2. Fuse both branches and compile the model with the Ada2Max optimizer.
3. Train on `train/` and validate on `test/` for the configured number of epochs.

## Model Architecture
| Component | Details |
|---|---|
| Patch embedding | Conv2D on MobileNet-V2 feature maps |
| Transformer encoder | `num_layers=18`, `num_heads=12`, `hidden_dim=60` |
| Secondary branch | VGG-19 (ImageNet weights, frozen) |
| Fusion | Concatenation of ViT CLS-token output and VGG-19 dense features |
| Output | Dense(1, sigmoid) |

## Results
> Fill in your accuracy / precision / recall / F1 / AUC table here, ideally with a comparison against baseline models reported in the paper.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| MobileNet-V2 (baseline) | — | — | — | — |
| VGG-19 (baseline) | — | — | — | — |
| **MLPVB (ours)** | — | — | — | — |

## Citation
If you use this code or find it helpful, please cite:

```bibtex
@article{yourlastname2026mlpvb,
  title   = {MLPVB: A Multi-Layer Perceptron Vision-Based Hybrid Technique for Posture-Based Heart Attack Detection},
  author  = {Your Name and Co-author Name},
  journal = {PeerJ Computer Science},
  year    = {2026},
  doi     = {10.7717/peerj-cs.XXXXX}
}
```

Also see [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata (GitHub renders a "Cite this repository" button automatically once this file is present).

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact
Maintained by **Your Name** ([[email protected]](mailto:[email protected])). Issues and pull requests are welcome.
