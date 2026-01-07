# Advanced EV Battery Health Prognostics

## 📌 Project Overview
This project implements a **Physics-Informed Hybrid Attention LSTM** model to predict the **State of Health (SoH)** and **Full-Charge Range (FCR)** of Electric Vehicle batteries. By combining deep learning with known physical constraints (monotonic degradation, knee-points), the system achieves high accuracy and reliability, suitable for Battery Management Systems (BMS).

## 🚀 Key Features
*   **Hybrid Attention LSTM:** Captures long-term dependencies and focuses on critical usage events.
*   **Physics-Informed Loss:** Penalizes predictions that violate physical laws (e.g., health increasing).
*   **Uncertainty Quantification:** Uses Monte Carlo Dropout to provide confidence intervals.
*   **Synthetic Data Generation:** Simulates realistic EV driving, charging, and environmental conditions.
*   **Explainability:** Incorporates SHAP analysis to identify key degradation drivers.

## 📂 Documentation
Detailed documentation is available in the `Documentation/` folder:

*   **[Technical Specification](Documentation/TECHNICAL_SPECIFICATION.md):** In-depth details on architecture, data pipelines, algorithms, and model hyperparameters.
*   **[User Guide](Documentation/USER_GUIDE.md):** Step-by-step instructions on installation, setup, usage, and configuration.

## 🛠️ Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib seaborn shap
    ```

2.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook advanced.ipynb
    ```

3.  **Follow the Steps:**
    *   **Generate Data:** Run the simulation cell to create `battery_data.csv`.
    *   **Train:** Execute training cells to optimize the LSTM model.
    *   **Evaluate:** View predictions, uncertainty plots, and error metrics.
  
4.  About the dataset:
   - this dataset is a hyper-realistic and accurate replica of the NASA Li-Ion battery data
   - inspired using multiple different versions of NASA data from kaggle (cleaned & un-cleaned both)

## 📊 Results Snapshot
*   **SoH Prediction Accuracy (R²):** > 0.99
*   **Main Degradation Drivers:** Charging Patterns, Driving Style, and Depth of Discharge (DoD).



