# Technical Specification - Advanced EV Battery Health Prediction

**Data Created:** 2026-01-07
**Last Modified:** 2026-01-07

## System Outline
The **Advanced EV Battery Health Prediction** project is a comprehensive AI-driven system designed to estimate the **State of Health (SoH)** and **Full-Charge Range (FCR)** of Electric Vehicle (EV) batteries. The system addresses the critical challenge of "range anxiety" and resale value assessment by leveraging deep learning to model non-linear degradation patterns while respecting physical constraints.

**High Level Objectives:**
*   Accurately predict SoH and FCR using historical usage data.
*   Incorporate physical laws into the training process (Physics-Informed Loss) to prevent unrealistic predictions (e.g., health increasing over time).
*   Provide explainability for predictions using SHAP analysis.
*   Quantify prediction uncertainty using Monte Carlo Dropout.

## Tech Stack
*   **Language:** Python 3.8+
*   **Deep Learning Framework:** PyTorch
*   **Data Manipulation:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn
*   **Machine Learning Utilities:** Scikit-learn (MinMaxScaler, metrics)
*   **Interpretability:** SHAP (SHapley Additive exPlanations)
*   **Environment:** Jupyter Notebook / PyCharm

## Architecture Diagram

```ascii
    +-------------------------------------------------------+
    |                   Input Sequence                      |
    | (Voltage, Current, Temp, DoD, Driving/Charging Ptrns) |
    |                   Shape: [Batch, Seq_Len, Features]   |
    +---------------------------+---------------------------+
                                |
                                v
    +-------------------------------------------------------+
    |                      LSTM Layer                       |
    |          Captures Long-Term Temporal Dependencies     |
    |                 Hidden Size: 128                      |
    +---------------------------+---------------------------+
                                |
                                v
    +-------------------------------------------------------+
    |                  Attention Mechanism                  |
    |        Computes Weights & Context Vector based on     |
    |                  Time-Step Importance                 |
    +---------------------------+---------------------------+
                                |
                                v
    +-------------------------------------------------------+
    |               Fully Connected Layers                  |
    |               FC1 (128 -> 64) + ReLU                  |
    |               Dropout (Regularization)                |
    |               FC2 (64 -> 2) [SoH, FCR]                |
    +---------------------------+---------------------------+
                                |
                                v
    +-------------------------------------------------------+
    |                     Final Output                      |
    |              Predicted SoH (%) & FCR (km)             |
    +-------------------------------------------------------+
```

## Data Source
*   **Type:** Synthetically Generated Data.
*   **Volume:** 15,000 Cycles.
*   **Generation Logic:** Simulated based on physics-based degradation formulas involving:
    *   **Driving Patterns:** 3 discrete modes (0, 1, 2) representing stress levels.
    *   **Charging Patterns:** 3 discrete modes.
    *   **Environmental Factors:** Temperature (Normal distribution around 25°C).
    *   **Usage Metrics:** Depth of Discharge (DoD) and Trip Length.
    *   **Degradation Factors:** Non-linear aging factors including "knee points" (accelerated degradation thresholds).

## Data Processing Workflow

1.  **Synthetic Data Generation:**
    *   Variables (Temp, DoD, Trip Length) are sampled from distributions.
    *   `SoH` is calculated iteratively: `SoH[t] = SoH[t-1] - degradation_step + noise`.
    *   `FCR` is derived from `SoH` and environmental factors.
    *   Data is saved to `battery_data.csv`.

2.  **Feature Engineering & Scaling:**
    *   **Physical Constraints Calculation:** A separate physics-based baseline (`SoH_phys`, `FCR_phys`) is calculated for the loss function.
    *   **Scaling:** `MinMaxScaler` is applied to features and targets to normalize data to [0, 1] range.

3.  **Sequence Creation:**
    *   **Sliding Window Approach:** Data is transformed into sequences of length `seq_len=10` to capture historical context for the LSTM.
    *   **Dataset Class:** Custom `EVBatteryDataset` handles retrieving `(features, targets, physics_constraints)`.

4.  **Train/Test Split:**
    *   80% Training, 20% Testing.

## ML Models & Algorithms

### Model: Hybrid Attention LSTM
**Reasoning:**
*   **LSTM:** Standard RNNs struggle with long sequences; LSTMs effectively handle time-series degradation traits.
*   **Attention:** Battery health often drops due to specific critical events (e.g., deep discharges). Attention mechanisms allow the model to focus on these crucial time steps rather than treating the entire history equally.

**Hyperparameters:**
*   `input_size`: 6 (Feature count)
*   `hidden_size`: 128
*   `num_layers`: 1
*   `dropout_prob`: 0.2
*   `output_size`: 2 (SoH, FCR)
*   `learning_rate`: 0.001
*   `epochs`: 180
*   `batch_size`: 32

**Training Logic:**
*   **Optimizer:** Adam.
*   **Loss Function:** Custom `physics_loss` combining:
    1.  **MSE Loss:** For prediction accuracy.
    2.  **FCR Penalty:** Ensures FCR predictions align with physics-based estimates.
    3.  **Monotonicity Penalty:** Enforces the constraint that SoH cannot increase over time (excluding minor noise).

**Evaluation Metrics:**
*   **Mean Absolute Error (MAE)**
*   **R² Score (Coefficient of Determination)**

**Model Performance (Test Set):**
*   **SoH:** R² ~ 0.9945, MAE ~ 0.6787
*   **FCR:** R² ~ 0.9414, MAE ~ 10.4734

**Limitations:**
*   **Synthetic Data:** The model is trained on simulated data. Real-world battery data contains more noise and complex electrochemical behaviors not fully captured here.
*   **Generalizability:** May require fine-tuning for different battery chemistries (e.g., LFP vs. NMC).

**Possible Improvements:**
*   Integration with real-world BMS (Battery Management System) datasets.
*   Deployment of the model to edge devices (e.g., microcontroller) using standard ONNX export.

## Config / Environment Requirements
*   **Python:** 3.8 or higher.
*   **Packages:** `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`.
*   **Hardware:** CPU is sufficient for inference; GPU (CUDA) recommended for faster training, though the model is lightweight.

## Error Handling
| Error / Issue | Probable Cause | Potential Fix |
| :--- | :--- | :--- |
| **CUDA Out of Memory** | Batch size too large for GPU VRAM. | Reduce `batch_size` in the code (e.g., from 32 to 16). |
| **Monotonicity Violation** | Model predicts Health increase. | The `physics_loss` lambda for monotonicity might be too low. Increase `lambda_mono`. |
| **Convergence Failure** | High learning rate or poor initialization. | Reduce `lr` (e.g., to 0.0001) or increase epochs. |
| **Missing Module Error** | Required libraries not installed. | Run `pip install -r requirements.txt` (or install individual packages). |

## Security Considerations
*   **Data Privacy:** If deployed, user driving data (trip length, patterns) must be anonymized.
*   **Model Robustness:** Adversarial attacks could theoretically manipulate input signals (e.g., Voltage sensors) to give false health readings. Input validation and sensor fusion are recommended for production.
