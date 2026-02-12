# SanskritGPT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A from-scratch **nanoGPT-style autoregressive transformer** implemented in TensorFlow CUDA for generating classical Sanskrit text. Explores LLM architectures on ancient, morphologically rich languages with complex grammar (Sandhi, Vibhakti, etc.).

Inspired by Andrej Karpathy's nanoGPT, this project builds decoder-only transformers step-by-step — from bigram/bag-of-words baselines to full multi-head self-attention.

## Key Features
- Progressive model implementations: bag-of-words → bigram → single-head attention → multi-head attention → positional encodings.
- Custom data loader for Devanagari script and Sanskrit corpus handling.
- GPU-accelerated training with TensorFlow CUDA.
- Experiment with non-English tokenization and generation on classical texts (Vedas, Upanishads, etc.).

## File Structure
- `train.py` — Main training script.
- `data_loader.py` — Loads and tokenizes Sanskrit corpus.
- `model_*.py` — Step-by-step model implementations (bigram, single/multi-head attention, etc.).
- `utils.py` — Helper functions.
- `requirements.txt` — Dependencies.

## Installation
```bash
git clone https://github.com/rakeshvar/SanskritGPT.git
cd SanskritGPT
pip install -r requirements.txt