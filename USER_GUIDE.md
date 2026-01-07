# User Guide - Advanced EV Battery Health Prediction

## Introduction
Welcome to the **Advanced EV Battery Health Prediction** system. This tool is designed for researchers, battery engineers, and data scientists working on Electric Vehicles (EVs). Its primary purpose is to simulate battery usage data and train a high-accuracy Artificial Intelligence (AI) model to predict the battery's **State of Health (SoH)** and **Full-Charge Range (FCR)**.

Who is this for?
*   **Researchers** studying battery degradation patterns.
*   **Students** learning about LSTM networks and Time-Series forecasting.
*   **Engineers** prototyping Battery Management System (BMS) algorithms.

## Key Features
*   **Synthetic Data Generator:** Creates realistic battery cycling data (15,000 cycles) including temperature, driving/charging patterns, and random noise.
*   **Physics-Aware AI:** Uses a Hybrid Attention LSTM model that learns from data but respects physical laws (e.g., batteries don't magically heal).
*   **Uncertainty Quantification:** Provides a "confidence score" for every prediction using Monte Carlo Dropout, critical for safety.
*   **Interpretable results:** Uses SHAP values to explain *why* a battery is degrading (e.g., "Is it the temperature or my driving style?").

## Installation & Setup Guide

### Prerequisites
*   **Python Interpreter:** Python 3.8 or newer.
*   **Package Manager:** `pip` or `conda`.
*   **IDE:** Jupyter Notebook, JupyterLab, or VS Code/PyCharm with Notebook support.

### Tools Required
Ensure you have the following Python libraries installed:
*   `torch` (PyTorch)
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `shap`

## Running the Application

### 1. Development Mode
To interact with the code, visualize plots, and retrain the model, run the project as a Jupyter Notebook.

**Command:**
```bash
jupyter notebook advanced.ipynb
```

### 2. Running Scripts
If you export the code to a python script (e.g., `main.py`), you can run it directly:

**Command:**
```bash
python main.py
```

## How to Use the System

The system is structured as a sequential workflow within the `advanced.ipynb` notebook. Follow these steps:

### Step 1: Data Generation
*   **Action:** Run the first code cell (Cell 14).
*   **Outcome:** The system generates 15,000 cycles of simulated battery data and saves it to `battery_data.csv`. You will see a confirmation message: "Data saved to 'battery_data.csv'".

### Step 2: Data Preprocessing & Physical Constraints
*   **Action:** Run Cell 15.
*   **Logic:** The system loads the CSV, calculates physics-based constraints (Physics-Informed Baseline), scales the data using `MinMaxScaler`, and prepares the PyTorch Dataset.
*   **Outcome:** Data is split into Training (80%) and Testing (20%) sets.

### Step 3: Model Training
*   **Action:** Run Cell 16.
*   **Logic:** Initializes the `HybridAttentionLSTM` model and trains it for 180 epochs.
*   **Outcome:** You will see the Training Loss decrease over epochs (e.g., "Epoch 30/180, Loss: 0.0032").

### Step 4: Inference with Uncertainty
*   **Action:** Run Cell 17.
*   **Logic:** The model executes inference using **Monte Carlo Dropout** (running the model multiple times with random dropout) to generate mean predictions and standard deviations (uncertainty).
*   **Outcome:** Predictions are stored for evaluation.

### Step 5: Evaluation & Visualization
*   **Action:** Run Cells 18 through 22.
*   **Outcome:**
    *   **Metrics:** MAE and R² scores are printed.
    *   **Tables:** A "Predicted Range Table" compares Targets vs. Predictions.
    *   **Plots:** Generates and saves plots like `advanced_soh_uncertainty.png`.
    *   **SHAP Analysis:** Displays the most important features driving degradation.

## Explain the Config

You can adjust the following variables directly in the code (Cell 14) to change the simulation environment:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `num_cycles` | `15000` | Total number of battery cycles to simulate. |
| `base_range` | `500` | The max range (km) of the EV when new. |
| `base_degradation_rate` | `0.0025` | How fast the battery degrades per cycle naturally. |
| `knee_threshold` | `70.0` | SoH percentage where degradation accelerates (Knee point). |
| `knee_severity` | `0.03` | Intensity of the degradation acceleration after knee point. |

## Directory Structure

```ascii
Research Project/
├── Documentation/
│   ├── TECHNICAL_SPECIFICATION.md
│   └── USER_GUIDE.md
├── advanced.ipynb          # Main Application Notebook
├── main.ipynb              # (Optional) Alternative/Legacy notebook
├── battery_data.csv        # Generated Dataset (Output of Step 1)
├── Project_Description.md  # High-level project summary
└── *.png                   # Generated plots (e.g., advanced_soh_uncertainty.png)
```

## Troubleshooting

| Issue | Cause | Potential Fix |
| :--- | :--- | :--- |
| **"ModuleNotFoundError: No module named 'torch'"** | PyTorch is not installed. | Run `pip install torch`. |
| **"CUDA out of memory"** | GPU memory is full. | In Cell 15, reduce `batch_size` from 32 to 16 or 8. |
| **Data file not found** | You skipped Step 1. | Ensure you run Cell 14 to generate `battery_data.csv` first. |
| **Plots not showing** | Matplotlib backend issue. | Add `%matplotlib inline` at the top of the notebook. |
