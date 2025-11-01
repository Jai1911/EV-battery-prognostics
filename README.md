# EV-battery-prognostics
Physics informed hybrid Bi-LSTM framework

Trustworthy AI for Battery Prognostics: A Hybrid Physics-Informed Framework

This repository contains the complete code for a research project on developing a trustworthy AI model for Electric Vehicle (EV) battery State-of-Health (SoH) prognostics. The project demonstrates a novel "Tri-Pillar" framework that integrates a state-of-the-art deep learning architecture with physics-informed constraints and Bayesian uncertainty quantification.

The central problem this project addresses is the "trust gap" in AI for safety-critical systems. While standard "black-box" models can be accurate, they are often brittle, uninterpretable, and over-confident. This work proposes a "glass-box" alternative that is not only accurate but also robust, confident, and interpretable.

1. Core Concept: The "Tri-Pillar" Framework
This project's methodology is built on three core pillars to ensure a trustworthy prognostic model:

Pillar 1: Accuracy (SOTA Architecture)

A state-of-the-art Bidirectional LSTM (BiLSTM) with an Attention mechanism is used as the core prognostic engine. This allows the model to capture complex, non-linear temporal dependencies from the battery's operational history.

Pillar 2: Robustness (Physics-Informed Priors)

A custom physics-informed loss function is implemented. This acts as a Bayesian prior, embedding a fundamental physical law (monotonic degradation, i.e., SoH(t) <= SoH(t-1)) as a "soft constraint." This prevents the model from making physically impossible predictions.

Pillar 3: Confidence (Bayesian Uncertainty)

Monte Carlo (MC) Dropout is used as a tractable Bayesian approximation. By running inference multiple times with active dropout, the model produces a distribution of predictions, allowing for the quantification of epistemic uncertainty (i.e., the model's own confidence in its prediction).

2. Key Features
Synthetic Data Generation: Includes a script to generate a high-fidelity, cumulative degradation dataset over 5,000+ cycles.

Model Benchmarking: Systematically compares a Pure SOTA (BiLSTM+Attn) model against the Hybrid SOTA (BiLSTM+Attn + Physics-Loss + MC Dropout) model.

Hyperparameter Tuning: Integrates Optuna for efficient optimization of model hyperparameters (learning rate, hidden size, layers).

Explainable AI (XAI): Implements SHAP and Attention visualizations to interpret the "black box" and validate that the model is learning an interpretable, physically-sound relationship.

Uncertainty Quantification: Generates predictive uncertainty bands (mean ± std) for robust, confidence-aware prognostics.
