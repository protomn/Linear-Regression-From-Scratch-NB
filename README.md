# Linear Regression from Scratch: A Comprehensive Implementation

A complete, production-ready implementation of Ordinary Least Squares (OLS) and Ridge Regression built from first principles in Python, with sklearn-validated accuracy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## **Project Overview**

This project implements linear regression from scratch, exploring multiple mathematical perspectives and solution methods including:
- Matrix algebra and optimization
- Singular Value Decomposition (SVD)
- Regularization theory
- Bayesian statistics
- Numerical linear algebra

**Key Achievement:** Matches sklearn's Ridge regression to machine precision (< 10⁻¹³ error) while being **5-6x faster**
**The increase in speed is attributed to skipping the overhead from sklearn's abstraction layers.**

---

## **Table of Contents**

- [Features](#-features)
- [Mathematical Foundation](#-mathematical-foundation)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Benchmarks](#-benchmarks)
- [Theory Deep Dive](#-theory-deep-dive)
- [Acknowledgments](#-acknowledgments)

---

## **Features**

### **OLS Regression**
- Normal equations with centered variables
- SVD-based solution for numerical stability
- Hat matrix computation and analysis
- Residual orthogonality verification
- R² and MSE metrics

### **Ridge Regression**
- **Four equivalent implementations:**
  - Primal formulation (normal equations)
  - SVD formulation (shrinkage view)
  - Augmented OLS (data augmentation trick)
  - Dual formulation (kernel-ready)
- Fast Leave-One-Out Cross-Validation (LOOCV)
- Automated lambda selection via grid search
- Bias-variance tradeoff analysis

### **Advanced Features**
- Bayesian interpretation (MAP estimator)
- Effective degrees of freedom
- Singular value shrinkage visualization
- Condition number analysis
- Production-validated against sklearn

---

## **Mathematical Foundation**

### **OLS Objective**
Minimize the sum of squared residuals:
```
min ||y - Xβ||²
 β
```

**Solution (Normal Equations):**
```
β̂ = (X'X)⁻¹X'y
```

**Solution (SVD):**
```
β̂ = VΣ⁻¹U'y    where X = UΣV'
```

### **Ridge Objective**
Add L2 penalty to prevent overfitting:
```
min ||y - Xβ||² + λ||β||²
 β
```

**Solution (Primal):**
```
β̂_ridge = (X'X + λI)⁻¹X'y
```

**Solution (SVD with Shrinkage):**
```
β̂_ridge = V·diag(σᵢ/(σᵢ² + λ))·U'y
```

**Bayesian Interpretation:**
```
Ridge = MAP estimate with Gaussian prior: β ~ N(0, τ²I)
where λ = σ²/τ²
```

---

## **Installation**

### **Requirements**
```bash
python >= 3.8
numpy >= 1.20
matplotlib >= 3.3
scikit-learn >= 0.24  # For benchmarking only
```

### **Install Dependencies**
```bash
pip install numpy matplotlib scikit-learn
```

### **Clone Repository**
```bash
git clone https://github.com/yourusername/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

---

## 🎬 **Quick Start**
```python
import numpy as np
from models import OLSRegressor, RidgeRegressor

# Generate sample data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples = 100, n_features = 5, noise = 10.0)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fit OLS
ols = OLSRegressor()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

# Fit Ridge
ridge = RidgeRegressor(lambda_ = 1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Compare
print(f"OLS Test MSE: {np.mean((y_test - y_pred_ols) ** 2):.4f}")
print(f"Ridge Test MSE: {np.mean((y_test - y_pred_ridge) ** 2):.4f}")
```

---

## **Usage Examples**

### **1. Basic OLS Regression**
```python
# Fit OLS
model = OLSRegressor()
model.fit(X_train, y_train)

# Get coefficients
print(f"Coefficients: {model.beta_}")
print(f"Intercept: {model.intercept_}")

# Make predictions
y_pred = model.predict(X_test)

# Analyze hat matrix
H = model.get_hat_matrix(X_train)
print(f"Hat matrix trace: {np.trace(H)}")  # Should equal number of features

# Check residual orthogonality
residuals = model.get_residuals(X_train, y_train)
```

### **2. Ridge with Multiple Methods**
```python
# Method 1: Normal equations (fastest)
ridge_normal = RidgeRegressor(lambda_ = 10.0)
ridge_normal.fit(X_train, y_train)

# Method 2: SVD (most stable)
ridge_svd = RidgeRegressor(lambda_ = 10.0)
ridge_svd.fit_svd(X_train, y_train)

# Method 3: Augmented OLS
ridge_aug = RidgeRegressor(lambda_ = 10.0)
ridge_aug.fit_augmented(X_train, y_train)

# Method 4: Dual formulation (kernel-ready)
ridge_dual = RidgeRegressor(lambda_ = 10.0)
ridge_dual.fit_dual(X_train, y_train)

# All give identical results!
print(np.allclose(ridge_normal.beta_, ridge_svd.beta_))  # True
```

### **3. Automated lambda Selection with LOOCV**
```python
# Grid search over lambda values
lambda_values = np.logspace(-2, 3, 50)
loocv_scores = []

for lam in lambda_values:
    model = RidgeRegressor(lambda_ = lam)
    model.fit(X_train, y_train)
    loocv_scores.append(model.loocv_mse(X_train, y_train))

# Find optimal lambda
optimal_lambda = lambda_values[np.argmin(loocv_scores)]
print(f"Optimal lambda: {optimal_lambda:.4f}")

# Fit final model
ridge_optimal = RidgeRegressor(lambda_ = optimal_lambda)
ridge_optimal.fit(X_train, y_train)
```

### **4. Degrees of Freedom Analysis**
```python
ridge = RidgeRegressor(lambda_ = 10.0)
ridge.fit(X_train, y_train)

# Compute effective degrees of freedom
df = ridge.degrees_of_freedom(X_train)
print(f"Effective df: {df:.2f} (out of {X_train.shape[1]} parameters)")

# Ridge reduces model complexity!
```

### **5. Bayesian Interpretation**
```python
# Ridge with λ = 5.0 corresponds to:
# Prior: β ~ N(0, σ²/5) where σ² is noise variance

lambda_val = 5.0
ridge = RidgeRegressor(lambda_ = lambda_val)
ridge.fit(X_train, y_train)

# Assuming s^2 = 1, prior variance tau^2 = 1/lambda
tau_squared = 1 / lambda_val
print(f"Implicit prior: β ~ N(0, {tau_squared:.3f})")
print(f"Prior std: {np.sqrt(tau_squared):.3f}")
```

---

## **API Reference**

### **OLSRegressor**

#### **Methods**

##### `__init__()`
```python
OLSRegressor()
```
Initialize OLS regressor.

##### `fit(X, y)`
```python
fit(X: np.ndarray, y: np.ndarray) -> OLSRegressor
```
Fit OLS model using normal equations.

**Parameters:**
- `X`: Training features (n_samples, n_features)
- `y`: Training targets (n_samples,)

**Returns:** self

##### `fit_svd(X, y)`
```python
fit_svd(X: np.ndarray, y: np.ndarray) -> OLSRegressor
```
Fit OLS using SVD decomposition (more numerically stable).

##### `predict(X)`
```python
predict(X: np.ndarray) -> np.ndarray
```
Make predictions on new data.

**Parameters:**
- `X`: Test features (n_samples, n_features)

**Returns:** Predictions (n_samples,)

##### `get_hat_matrix(X)`
```python
get_hat_matrix(X: np.ndarray) -> np.ndarray
```
Compute hat matrix H = X(X'X)⁻¹X'.

**Returns:** Hat matrix (n_samples, n_samples)

##### `get_residuals(X, y)`
```python
get_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray
```
Compute residuals and verify orthogonality.

**Returns:** Residuals (n_samples,)

#### **Attributes**

- `beta_`: Coefficient vector (n_features,)
- `intercept_`: Intercept term (scalar)
- `X_mean_`: Training feature means (n_features,)
- `y_mean_`: Training target mean (scalar)
- `condition_number_`: Condition number of X (if using SVD)

---

### **RidgeRegressor**

#### **Methods**

##### `__init__(lambda_ = 1.0)`
```python
RidgeRegressor(lambda_: float = 1.0)
```
Initialize Ridge regressor.

**Parameters:**
- `lambda_`: Regularization strength (lambda ≥ 0). lambda = 0 recovers OLS.

##### `fit(X, y)`
```python
fit(X: np.ndarray, y: np.ndarray) -> RidgeRegressor
```
Fit Ridge using normal equations: β = (X'X + λI)⁻¹X'y

##### `fit_svd(X, y)`
```python
fit_svd(X: np.ndarray, y: np.ndarray) -> RidgeRegressor
```
Fit Ridge using SVD with shrinkage factors.

##### `fit_augmented(X, y)`
```python
fit_augmented(X: np.ndarray, y: np.ndarray) -> RidgeRegressor
```
Fit Ridge as OLS on augmented data [X; √λI].

##### `fit_dual(X, y)`
```python
fit_dual(X: np.ndarray, y: np.ndarray) -> RidgeRegressor
```
Fit Ridge using dual formulation (kernel-ready).

##### `predict(X)` / `predict_dual(X)`
```python
predict(X: np.ndarray) -> np.ndarray
```
Make predictions (use predict_dual if fitted with fit_dual).

##### `loocv_mse(X, y)`
```python
loocv_mse(X: np.ndarray, y: np.ndarray) -> float
```
Compute Leave-One-Out CV MSE using fast closed-form formula.

**Returns:** LOOCV MSE (scalar)

##### `dof(X)`
```python
degrees_of_freedom(X: np.ndarray) -> float
```
Compute effective degrees of freedom: df(λ) = Σ σᵢ²/(σᵢ² + λ)

**Returns:** Effective df (scalar)

#### **Attributes**

Same as OLSRegressor, plus:
- `singular_values_`: Singular values of X (if using SVD)
- `shrinkage_factors_`: Shrinkage applied to each direction (if using SVD)
- `alpha_`: Dual variables (if using dual formulation)
- `X_train_centered_`: Stored training data (for dual predictions)

---

## **Benchmarks**

### **Accuracy Comparison (Boston Housing Dataset)**

Tested on 506 samples, 13 features:

| λ      | Your MSE | sklearn MSE | Difference  | Verdict |
|--------|----------|-------------|-------------|---------|
| 0.01   | 24.2917  | 24.2917     | 1.42×10⁻¹⁴  | Perfect |
| 0.1    | 24.3010  | 24.3010     | 7.11×10⁻¹⁵  | Perfect |
| 1.0    | 24.4772  | 24.4772     | 1.07×10⁻¹⁴  | Perfect |
| 10.0   | 24.6483  | 24.6483     | 1.07×10⁻¹⁴  | Perfect |
| 100.0  | 23.4659  | 23.4659     | 7.11×10⁻¹⁵  | Perfect |

**Maximum coefficient difference:** < 10⁻¹³ (machine precision!)

### **Speed Comparison**

| Method | Your Time | sklearn Time | Speedup |
|--------|-----------|--------------|---------|
| Ridge  | 0.0002s   | 0.0012s      | **6.0x** |
| OLS    | 0.0006s   | 0.0015s      | **2.5x** |

**Why faster?** Streamlined implementation without sklearn's overhead.

---

## **Theory Deep Dive**

### **1. Why Ridge Regression?**

**Problem with OLS:**
- High variance when features are correlated
- Unstable when p ≈ n or condition number is large
- Overfits on small datasets

**Ridge Solution:**
- Adds penalty λ||β||² to encourage small coefficients
- Reduces variance at cost of small bias
- Improves test performance via bias-variance tradeoff

### **2. The Four Perspectives**

#### **Perspective 1: Optimization**
```
Ridge = arg min ||y - Xβ||² + λ||β||²
          β
```
Penalized least squares with L2 regularization.

#### **Perspective 2: SVD Geometry**
```
β̂_ridge = Σᵢ (σᵢ/(σᵢ² + λ)) · (uᵢ'y) · vᵢ
```
Each singular value direction is *shrunk* by factor σᵢ²/(σᵢ² + λ).
- Large σᵢ (strong signal) → minimal shrinkage
- Small σᵢ (weak signal) → heavy shrinkage

#### **Perspective 3: Bayesian Inference**
```
Prior:      β ~ N(0, τ²I)
Likelihood: y|β ~ N(Xβ, σ²I)
Posterior:  β|y ~ N(β̂_ridge, Σ)

where λ = σ²/τ²
```
Ridge = Maximum A Posteriori (MAP) estimate with Gaussian prior.

#### **Perspective 4: Augmented Data**
```
Ridge on (X, y) = OLS on (X̃, ỹ) where:
X̃ = [X; √λI]
ỹ = [y; 0]
```
"Pseudo-observations" pull coefficients toward zero.

### **3. Key Insights**

**Condition Number:**
```
κ(X) = σ_max / σ_min
```
- κ close to 1: Well-conditioned, OLS works fine
- κ > 100: Ill-conditioned, Ridge essential
- κ > 10,000: Severe multicollinearity

**Degrees of Freedom:**
```
df(λ) = trace(H_ridge) = Σ σᵢ²/(σᵢ² + λ)
```
- df(0) = p (OLS uses all parameters)
- df(∞) = 0 (extreme ridge suppresses all)
- df(λ) smoothly interpolates

**Optimal λ Selection:**
- Too small: Overfitting (high variance)
- Too large: Underfitting (high bias)
- Use cross-validation to balance

---

## **Acknowledgments**

### **Educational Resources**
- *Elements of Statistical Learning* - Hastie, Tibshirani, Friedman
- *Pattern Recognition and Machine Learning* - Christopher Bishop
- *Convex Optimization* - Boyd & Vandenberghe
- Stanford CS229 (Machine Learning) - Andrew Ng

### **Mathematical Foundations**
- SVD theory and applications
- Regularization theory (Tikhonov)
- Bayesian inference
- Matrix calculus

### **Implementation Inspiration**
- scikit-learn documentation and source code
- NumPy's numerical linear algebra
- Academic papers on ridge regression

---

## **Contact**

**Author:** [Pratham nayak]  
**GitHub:** [@protom](https://github.com/protomn)  

If you found this project helpful, please consider giving it a star! ⭐

---

## **Citation**

If you use this code in your research, please cite:
```bibtex
@software{linear_regression_from_scratch,
  author = {Your Name},
  title = {Linear Regression from Scratch: A Comprehensive Implementation},
  year = {2025},
  url = {https://github.com/yourusername/linear-regression-from-scratch}
}
```

---

## **Learning Outcomes**

After working through this project, I understood:

**Mathematical Theory**
- Normal equations and their derivation
- Singular Value Decomposition (SVD)
- Regularization and bias-variance tradeoff
- Bayesian interpretation of ridge regression
- Effective degrees of freedom

**Numerical Methods**
- Stable matrix inversion techniques
- When to use SVD vs normal equations
- Numerical precision and conditioning
- Efficient cross-validation

**Data Science Skills**
- Model selection and hyperparameter tuning
- Cross-validation strategies
- Benchmarking against standard libraries
- Visualization of complex concepts

---

**Built with Numpy and Python**
