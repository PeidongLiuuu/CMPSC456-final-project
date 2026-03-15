# CMPSC 456 – Numerical Interpolation Study on Housing Price Data

**Author:** Peidong Liu
**Course:** CMPSC 456
**Dataset:** [Housing Price Data – Kaggle](https://www.kaggle.com/datasets/saurabhbadole/housing-price-data)

---

## Project Overview

This project investigates polynomial interpolation methods applied to real housing price data, examining the trade-offs between **equispaced** and **Chebyshev** nodes, the emergence of **Runge-type oscillations**, and the behavior of **cubic splines** under different boundary conditions. Errors are analyzed using both the **L2** and **L∞** (minimax) norms.

---

## Files

| File | Description |
|------|-------------|
| `project_code.py` | Main experiment script |
| `Housing_Price_Data.csv` | Kaggle housing dataset (545 rows, 13 features) |
| `plot1_equi_vs_cheb.png` | Equispaced vs Chebyshev interpolation curves & errors |
| `plot2_runge_analysis.png` | L2 and L∞ error vs polynomial degree |
| `plot3_runge_visual.png` | Runge oscillation visualization at degree 19 |
| `plot4_splines.png` | Cubic spline comparison (Natural, Clamped, Not-a-Knot) |
| `plot5_error_comparison.png` | L2 vs L∞ bar chart across all methods |

---

## Experiments

### 1. Equispaced vs Chebyshev Barycentric Interpolation (`plot1_equi_vs_cheb.png`)

Both node types are evaluated using the **barycentric form** for numerical stability.
At n=15 nodes:

| Method | L∞ Error | L2 Error |
|--------|----------|----------|
| Equispaced | 8.3501 | 1.5317 |
| Chebyshev  | 0.9487 | 0.2316 |

**Key finding:** Chebyshev nodes dramatically reduce interpolation error by concentrating nodes near the endpoints, counteracting Runge-type oscillations.

---

### 2. Runge-Type Oscillation Analysis (`plot2_runge_analysis.png`, `plot3_runge_visual.png`)

Error is tracked as polynomial degree increases from 4 to 20.

| Degree | Equispaced L∞ | Chebyshev L∞ |
|--------|--------------|--------------|
| 4      | 1.05         | 1.10         |
| 10     | 6.61         | 1.16         |
| 18     | 67.64        | 0.87         |
| 20     | 281.73       | 1.12         |

**Key finding:** Equispaced nodes exhibit severe Runge oscillations at high degree (L∞ error exceeds 281 at n=20), while Chebyshev nodes remain stable and even improve with degree.

---

### 3. Cubic Spline Comparison (`plot4_splines.png`)

Three boundary conditions are compared on the same dataset:

| Spline Type | L∞ Error | L2 Error |
|-------------|----------|----------|
| Natural (2nd deriv = 0) | 0.6695 | 0.2113 |
| Clamped (slope = 0)     | 0.6695 | 0.1941 |
| Not-a-Knot              | 2.0907 | 0.8657 |

**Key finding:** Natural and Clamped splines perform nearly identically in L∞ but Clamped achieves a slightly lower L2 error. Not-a-Knot performs worst here, likely due to the dataset's behavior near the endpoints.

---

### 4. L2 vs L∞ Error Comparison (`plot5_error_comparison.png`)

Summary across all methods:

| Method | L∞ Error | L2 Error |
|--------|----------|----------|
| Equispaced Bary (n=15) | 8.3501 | 1.5317 |
| Chebyshev Bary (n=15)  | 0.9487 | 0.2316 |
| Natural Spline          | 0.6695 | 0.2113 |
| Clamped Spline          | 0.6695 | 0.1941 |
| Not-a-Knot Spline       | 2.0907 | 0.8657 |

**Key finding:** Cubic splines (Natural/Clamped) achieve the best overall accuracy. L∞ captures worst-case oscillation behavior while L2 reflects average accuracy — these can diverge significantly for methods prone to Runge oscillations.

---

## Setup & Usage

```bash
# Install dependencies
pip install numpy pandas matplotlib scipy kagglehub

# Download dataset (or use the CSV already in this directory)
python3 -c "import kagglehub, shutil, os; p=kagglehub.dataset_download('saurabhbadole/housing-price-data'); [shutil.copy2(os.path.join(p,f),'.') for f in os.listdir(p)]"

# Run all experiments
python3 project_code.py
```

---

## Research Questions Addressed

1. **Equispaced vs Chebyshev:** Chebyshev nodes consistently outperform equispaced nodes, especially at higher degrees.
2. **Runge oscillations:** Clearly observable with equispaced nodes — error grows exponentially with degree. Chebyshev nodes are immune.
3. **Cubic splines:** All spline variants outperform high-degree polynomial interpolation. Natural and Clamped boundary conditions yield the most stable results on this dataset.
