
# Quantum-Inspired Neural Networks (QINN)

**Author:** Ghanta Krishna Murthy  
**Domain:** Machine Learning, Quantum-Inspired AI  
**Hardware:** Classical GPU (No Quantum Hardware)

---

## Motivation
Quantum Machine Learning promises computational advantages but is currently
limited by noisy, scarce quantum hardware. This work explores whether
*quantum-inspired inductive biases* can be embedded into **classical neural
networks**, enabling improved learning behavior without requiring quantum
devices.

---

## Core Idea
We abstract three principles from quantum mechanics:

- **Superposition** → multiple representations learned simultaneously  
- **Phase** → additional degrees of freedom beyond magnitude  
- **Interference** → constructive and destructive feature interactions  

These principles are implemented using **real-valued, differentiable
operations**, making the approach fully compatible with standard deep learning
pipelines.

---

## Architecture
The proposed Quantum-Inspired Layer (QIL) decomposes features into:

- **Amplitude branch** (feature strength)
- **Phase branch** (interference control)

The output is computed as:
output = amplitude × cos(phase)


This introduces phase-sensitive interference while remaining GPU-efficient.

---

## Experiments

### Baseline Comparison
- Dataset: MNIST
- Baseline: Classical MLP
- Proposed: MLP + Quantum-Inspired Layer

**Result:**  
QINN converges faster and achieves comparable final accuracy.

---

### Low-Data Regime (10% Training Data)
QINN consistently outperforms the baseline by **2–3% test accuracy**, indicating
improved **sample efficiency**.

---

### Robustness to Noise
We evaluate performance under increasing Gaussian noise.

![Accuracy vs Noise](results/accuracy_vs_noise.png)

**Observation:**
- QINN shows higher sensitivity to strong noise
- Indicates a trade-off between representational sharpness and robustness

---

## Key Findings
- Quantum-inspired inductive biases improve learning efficiency under limited data
- Phase-based interference accelerates early convergence
- Increased expressivity introduces noise sensitivity
- Trade-offs are explicit, measurable, and controllable

---

## Conclusion
This project demonstrates that **quantum-inspired principles can be practically
useful on classical hardware**, offering benefits in learning efficiency while
revealing important robustness trade-offs.

The results motivate future work on **phase regularization and adaptive
interference control**.

---

## Reproducibility
All experiments are fully reproducible using Kaggle notebooks and standard
PyTorch workflows.
