complete_readme_content = """# Bearing Fault Diagnostics & Domain Adaptation Pipeline

## 1. System & Workspace Environment
This entire project and machine learning pipeline was developed, debugged, and verified on a **Pop!_OS Linux** workstation. To maintain dependency isolation and prevent path contamination, the toolchain runs inside a dedicated python virtual environment.

### Workspace Specifications
* **Active Working Directory:** `~/vs_code/minor.py/`
* **Python Runtime Environment:** Version 3.12.x (`./venv/bin/python3`)
* **Package Installer Core:** `./venv/bin/python3 -m pip`

### Project Dependencies
* **numpy**: Handles vector calculations and handles data serialization/loading for raw metric matrices (.npy files).
* **pandas**: Handles tabular DataFrames and performance summary structuring.
* **scikit-learn**: Powers the baseline model (RandomForestClassifier), data fold splitting (StratifiedGroupKFold), and non-linear feature maps (TSNE).
* **matplotlib**: Script plotting engine used to render multi-class ROC validation paths and t-SNE scatter configurations.
* **joblib**: Serializes and dumps heavy, fully trained model binaries out to disk.
* **openpyxl**: Interacts with Excel worksheets directly to inject clean font alignments and weights without external markup.
* **jinja2**: Internal layout dependency required to compile multi-index styling configurations.

---

## 2. Pipeline Architecture & Execution Flow
The system acts as a sequential four-stage machine learning orchestration pipeline executing through a single entry point (main.py).

### Stage 1: Preprocessing & Alignment
Extracts high-dimensional signal features from the raw inputs and syncs domain groups. Class labels are verified across both datasets using a central blueprint mapping configuration file located at data/class_mapping.json.
* **10 Fault Classes Managed:** Healthy Condition, Inner Race (IR) Minor/Moderate/Severe, Outer Race (OR) Minor/Moderate/Severe, and Ball Minor/Moderate/Severe.

### Stage 2: Source Domain Training (CWRU Data)
Handled inside src/ml/step05_training_and_evaluation.py. A Random Forest classifier is initiated with a fixed structural architecture to restrict overfitting.
* **Airtight Pipeline Seeding:** Complete reproducibility across script executions is locked in by applying a uniform seed configuration. The model relies on random_state=9, and the cross-validation routing is locked using StratifiedGroupKFold with shuffle=True and random_state=9. This prevents variations in evaluation metrics between code executions.

### Stage 3: Target Domain Cross-Inference (ANSYS Data)
Handled inside src/ml/step06_evaluate_ansys.py. The frozen model weights (model/random_forest_model.joblib) are loaded to infer predictions against the completely unseen target domain features (X_ansys).
* **Path & Overwrite Fixes:** All output image files and plot labels are isolated. Target graphs output directly to ansys_roc_auc_curve.png using the custom title "Ansys Multi-Class ROC Curves", ensuring that prior source-domain chart exports are never overwritten. No seed is required here as model.predict() is a purely deterministic forward pass.

### Stage 4: Domain Analysis & Distribution Mapping
Handled inside src/ml/step07_domain_comparision.py. The cross-domain performance degradation is calculated and exported across two primary data outputs.

---

## 3. Pipeline Outputs & Structural Interpretation

### Output A: Multi-Class Confusion Matrices (cwru_confusion_matrix.png & ansys_confusion_matrix.png)
The pipeline automatically generates and plots two distinct multi-class confusion matrices to visualize the exact true vs. predicted class alignments.

#### Analytical Interpretation:
* **CWRU Confusion Matrix:** Represents the validation performance on real-world data under cross-validation splits. It shows high diagonal concentration, proving that the model successfully differentiates complex fault types (Minor vs. Severe) with minimal off-diagonal misclassifications.
* **ANSYS Confusion Matrix:** Represents cross-domain inference behavior. Off-diagonal dispersion reveals where the domain gap causes the model to stumble. For instance, minor simulation artifacts can lead the model to misclassify "IR Minor" as "Healthy" or confuse adjacent fault severities due to shifting acoustic and vibration distributions between real and synthetic testing.

### Output B: Cross-Domain F1 Comparative Summary (f1_comparison_table.xlsx)
A spreadsheet that structures the cross-domain results. It features an explicitly bolded first row (Column Headers) and an explicitly bolded final row (Calculated Metrics Average) formatted down to exactly 4 decimal places.

| **Fault Class** | **F1 CWRU** | **F1 Ansys** | **F1 Drop** |
| Healthy         |    0.9950   |    0.9210    |   0.0740    |
| IR Minor        |    0.9810   |    0.8430    |   0.1380    |
| ... [Data Rows] |     ...     |      ...     |     ...     |
| **Average**     |  **0.9664** |  **0.8065**  |  **0.1599** |

#### Analytical Interpretation:
* **F1 CWRU:** Represents the baseline performance under ideal conditions.
* **F1 Ansys:** Measures the model's out-of-the-box generalization capability.
* **F1 Drop:** Quantifies the impact of the mathematical domain shift. Larger drop values indicate specific fault categories where numerical simulations differ significantly from real-world physics.

### Output C: Multi-Class Latent Distribution Space Map (domain_tsne_plot.png)
High-dimensional feature vectors from both domains are stacked together and compressed into a 2D space using a fixed seed (random_state=9) and a local neighborhood perplexity of 30.
* **Visual Protocol:** The real-world verification points (CWRU) are plotted as Solid Dots. The numerical simulation features (ANSYS) are overlaid as Hollow Circles using matching category colors across a high-contrast 10-class palette.
#### Analytical Interpretation: If hollow circles and solid dots of the same color form overlapping clusters, the simulation has captured the real-world feature distribution accurately. Spatial separation or detached clusters point to a domain gap, highlighting where the simulation model needs refinement.

---

## 4. Execution Commands
To execute the complete end-to-end processing pipeline from scratch, open your terminal, navigate to the workspace, and run:

```bash
# Activate your local isolated virtual environment wrapper
source venv/bin/activate

# Execute the master orchestration file
python3 main.py