# Installation Guide

This guide explains how to set up and run the **Quantum-Inspired Neural Networks (QINN)** project.

---

## 1. Clone the Repository
```bash
git clone https://github.com/kittu2217/quantum-inspired-neural-networks.git
cd quantum-inspired-neural-networks
```

## 2.Create a Virtual Environment(Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate   # Windows
```
## 3.Install Dependencies
```bash
pip install -r requirements.txt
```
## 4.Run the end-to-end experiment
```bash
python run_mnist_experiment.py
```
## 5.Reproducibility
All experiments are deterministic by default using a fixed random seed
configured in config.yaml.

## 6.Tests
```bash
pytest
```
## Notes
- GPU is optional but recommended
- The project is compatible with Kaggle and Google Colab
