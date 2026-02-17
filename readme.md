# Tax Evasion Simulation & Analysis

This project simulates income tax evasion behavior under varying levels of tax progressivity ($\beta$) and evasion heterogeneity ($\sigma$). It provides a framework for analyzing the gap between "True" and "Reported" income inequality statistics.

## 🚀 Overview
The simulation engine models how reporting behavior shifts measured inequality, allowing researchers to decompose the "Inequality Gap" into measurement error and agent re-ranking effects.

### Key Technical Features:
* **Memory Optimization**: Specifically tuned to handle populations of up to 5,000,000 agents on a 16GB RAM system by utilizing efficient NumPy arrays and on-demand Pandas conversion.
* **Calibrated Scenarios**: Includes automated bisection solvers to match target reported income shares (e.g., Top 1% share).
* **Robustness**: Supports Lognormal and Pareto distributions with both Multiplicative (Log-linear) and Additive evasion modes.

## 📂 Project Structure
* `tax_model.py`: The core engine containing distribution generators, evasion functions, and calibration solvers.
* `run_analysis.py`: The master driver script that pre-computes simulation grids and generates all outputs.

## 📊 List of Outputs
When you run `run_analysis.py`, the following files are generated in your project folder:

| File | Description |
| :--- | :--- |
| **Fig1_EvasionRates.pdf** | Heatmap of evasion rates for the Top 1% and 0.1%. |
| **Fig2_TaxGap.pdf** | Heatmap of the aggregate tax gap. |
| **Fig3_ReportedGap.pdf** | Heatmap of the "Inequality Gap" (True - Reported). |
| **Fig4_ShareLines.pdf** | Comparison of Reported vs. True shares (Top 10% to 0.1%). |
| **Fig5_EvasionProfiles.pdf** | Curves showing evasion rates vs. income levels. |
| **Fig6_Robustness_Additive.pdf**| Robustness check using Additive evasion logic. |
| **Fig7_Robustness_Pareto.pdf** | Robustness check using Pareto distributions. |
| **Fig_Walkthrough_Clean.pdf** | Summary density plot and selection effect analysis. |
| **Fig_GiniGap.pdf** | Heatmap of the difference between True and Reported Gini. |
| **Tab1_Decomposition.csv** | Table showing Measurement vs. Re-ranking effects. |
| **Fig_Robustness_FixedTrue_Gap.pdf** |Heatmap of reported income gap holding true fixed |
| **Fig_FixedTrue_Robustness.pdf** | Heatmap of reported 1% shares, holding true fixed |

## ⚙️ Setup & Requirements
You need Python 3.x installed along with the following libraries:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

Install them via command line:
```bash

pip install numpy pandas matplotlib seaborn scipy
