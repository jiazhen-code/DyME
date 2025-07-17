## DyME

### Empowering Small VLMs 🧠 with Dynamic Memorization & Exploration
> **“Empowering Small VLMs to Think with Dynamic Memorization and Exploration”**  
> Jian‑Liang Liu *et al.* [[arXiv 2506.23061](https://arxiv.org/abs/2506.23061)]

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![PyTorch ≥ 2.1](https://img.shields.io/badge/pytorch-≥2.1-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-transformers-red.svg)](https://github.com/huggingface/transformers)

---

### ✨ Highlights
* **Dynamic Memory Module** – differentiable read/write slots let small VLMs cache and reuse visual contexts.
* **Exploration Controller** – learns when to query memory vs. external knowledge for better reasoning.
* **Lightweight** – fits on a single 8 GB GPU for inference; full training needs one A100 or four 24 GB GPUs.

---

## 📂 Repository Layout
```

.
├── src/       
├── script/      
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation
```bash
git clone 
cd DyME

# create env (conda recommended)
conda create -n DyME python=
conda activate DyME

# install dependencies
pip install -r requirements.txt
# or: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````

---

## 🚀 Quick Start

### Inference

```python
from PIL import Image


```

### Finetuning

```bash
python 
```


---

## 🔬 Results



---

## 📖 Citation

```bibtex
@article{liu2025empowering,
  title={Empowering Small VLMs to Think with Dynamic Memorization and Exploration},
  author={Liu, Jiazhen and Deng, Yuchuan and Chen, Long},
  journal={arXiv preprint arXiv:2506.23061},
  year={2025}
}
```

---

## 🤝 Contributing

---

## 📝 License

