# Phase 1: Statistical Foundations & Optimization (Days 1–5)

## Overview
This week transitions from the "black box" usage of libraries to mastering the mathematical mechanics of linear models and the optimization algorithms that power them.

---

## Day 1: The Analytical Solution (Normal Equations)
*Goal: Understand when an exact solution is possible and its computational cost.*

- **Topics:** - Ordinary Least Squares (OLS) derivation.
    - Matrix calculus: Deriving $\nabla_{\mathbf{w}} \|\mathbf{Xw} - \mathbf{y}\|^2$.
    - The Normal Equation: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.
- **Mastery Task:** - Implement a `LinearRegression` class using only `NumPy`.
    - Use `np.linalg.solve` vs `np.linalg.inv` and compare numerical stability.
    - **Discussion:** Why does $O(n^3)$ complexity make this unfeasible for "Big Data"?

---

## Day 2: First-Order Optimization (Gradient Descent)
*Goal: Transition from analytical solutions to iterative optimization.*

- **Topics:**
    - The Loss Surface: Convexity of MSE.
    - Gradient descent update rule: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$.
    - Learning rates ($\eta$) and the vanishing/exploding gradient problem.
- **Mastery Task:**
    - Manually compute the Jacobian for a simple linear layer.
    - Implement Vanilla Gradient Descent and plot the "Loss vs. Iteration" curve.
    - Visualize the weight trajectory over a 2D contour plot of the loss function.

---

## Day 3: Stochasticity and Mini-Batching
*Goal: Bridge the gap between theoretical GD and industry-standard training.*

- **Topics:**
    - Batch GD vs. Stochastic GD (SGD) vs. Mini-batch GD.
    - The "Noise-Ball" effect: Why SGD never truly settles at the minimum.
    - Learning rate schedulers: Step Decay, Exponential Decay, and Warm-up.
- **Mastery Task:**
    - Implement a data loader that handles shuffling and mini-batch generation.
    - Compare convergence speed (time-to-accuracy) between Batch GD and Mini-batch GD on a dataset with $10^5$ samples.

---

## Day 4: Logistic Regression & GLMs
*Goal: Moving beyond regression to classification via Maximum Likelihood Estimation.*

- **Topics:**
    - The Sigmoid/Logit link function.
    - Bernoulli distribution and the Binary Cross-Entropy (BCE) loss.
    - **The Hessian:** Why we can't solve Logistic Regression analytically (Iterative Reweighted Least Squares).
- **Mastery Task:**
    - Implement the Logistic Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
    - Write a custom training loop for Binary Classification.
    - **Challenge:** Implement the Softmax function for multi-class classification and derive its gradient.



---

## Day 5: Regularization & The Geometry of Sparsity
*Goal: Constraining the hypothesis space to prevent overfitting.*

- **Topics:**
    - $L_2$ Regularization (Ridge): The weight decay perspective.
    - $L_1$ Regularization (Lasso): The Laplacian prior and feature selection.
    - Elastic Net: Balancing $L_1$ and $L_2$.
- **Mastery Task:**
    - Implement Ridge and Lasso by modifying your Day 2 Gradient Descent code.
    - **Visual Experiment:** Create a plot showing how increasing the $\lambda$ (penalty) in Lasso drives coefficients exactly to zero, whereas in Ridge, they only approach zero.

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Libraries:** `NumPy`, `Matplotlib`, `SciPy` (for optimization verification).
- **Core Concept:** Avoid `scikit-learn` this week. Focus on vectorization ($\mathbf{X} \cdot \mathbf{w}$) rather than for-loops.


# Phase 1: Statistical Foundations & Classical ML (Days 6–10)

## Overview
This week focuses on non-linear decision boundaries and the power of ensembling. You will move from single-node logic to complex gradient-boosted architectures, focusing on the trade-offs between bias and variance.

---

## Day 6: Information Theory & The Root Node
*Goal: Understand the mathematical criteria for splitting data.*

- **Topics:**
    - **Shannon Entropy:** $H(S) = -\sum p_i \log_2 p_i$.
    - **Information Gain:** The reduction in entropy after a split.
    - **Gini Impurity:** $1 - \sum p_i^2$ (The CART standard).
- **Mastery Task:**
    - Write a function `calculate_gini()` and `calculate_entropy()` from scratch.
    - Create a script that iterates through a single feature and finds the "Optimal Split Point" by minimizing the weighted impurity of the resulting children.

---

## Day 7: Building the Recursive CART
*Goal: Implement a recursive algorithm for a fully grown Decision Tree.*

- **Topics:**
    - Recursive binary splitting.
    - Stopping Criteria: `max_depth`, `min_samples_split`, and `min_impurity_decrease`.
    - Handling Numerical vs. Categorical features.
- **Mastery Task:**
    - Implement a `DecisionTree` class with a `fit` method that recursively builds a tree structure (nodes and leaves).
    - Implement a `predict` method that traverses the tree for a given input vector.



---

## Day 8: Bagging & Random Forests
*Goal: Reducing variance through bootstrap aggregation.*

- **Topics:**
    - **Bootstrap Sampling:** Sampling with replacement.
    - **Feature Bagging (The "Random" in Random Forest):** Selecting a random subset of $\sqrt{d}$ features at each split.
    - **Out-of-Bag (OOB) Error:** Using unsampled data for internal validation.
- **Mastery Task:**
    - Build a `RandomForest` class that aggregates multiple `DecisionTree` instances from Day 7.
    - Implement the "Majority Vote" mechanism for classification.
    - **Experiment:** Compare the variance of a single deep tree vs. a 100-tree forest on a noisy dataset.

---

## Day 9: Adaptive Boosting (AdaBoost)
*Goal: The philosophy of "Focusing on the Hard Samples."*

- **Topics:**
    - **Decision Stumps:** Trees with depth = 1.
    - **Sample Weighting:** Exponentially increasing weights for misclassified points.
    - **Alpha ($\alpha$):** Calculating the "Amount of Say" for each weak learner.
- **Mastery Task:**
    - Implement the AdaBoost update loop.
    - Manually update sample weights $w_i \leftarrow w_i \cdot \exp(\alpha \cdot \mathbb{I}(y_i \neq G(x_i)))$.
    - Visualize how the decision boundary shifts to encompass previously misclassified outliers.



---

## Day 10: Gradient Boosting Mechanics
*Goal: Optimizing loss functions in the function space.*

- **Topics:**
    - **Residual Learning:** Fitting the next tree to the *negative gradient* of the loss function.
    - **Learning Rate (Shrinkage):** The role of $\nu$ in preventing overfitting.
    - **XGBoost/LightGBM Enhancements:** Brief overview of Second-order Taylor expansion (Hessians) and Histogram-based splitting.
- **Mastery Task:**
    - Implement a simple **Gradient Boosted Regressor**.
    - Initialize with the mean value, then iteratively add trees that predict the residuals: $r_{im} = -[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}]_{f=f_{m-1}}$.

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Key Concept:** Recursive programming and the Bias-Variance Tradeoff.
- **Visualization:** Use `Graphviz` to export and view your custom-built tree structures.


# Phase 1: Statistical Foundations & Classical ML (Days 11–15)

## Overview
This week explores how to handle high-dimensional feature spaces. You will master linear dimensionality reduction (PCA), understand the manifold hypothesis (t-SNE), and dive deep into the constrained optimization theory behind Support Vector Machines.

---

## Day 11: PCA & Linear Manifolds
*Goal: Master Eigen-decomposition and Variance Maximization.*

- **Topics:**
    - **Change of Basis:** Projecting data onto orthogonal vectors.
    - **The Covariance Matrix:** $\Sigma = \frac{1}{n-1}X^T X$.
    - **Eigen-decomposition:** Solving $Av = \lambda v$ to find Principal Components.
- **Mastery Task:**
    - Implement PCA using `numpy.linalg.eig`.
    - Calculate the **Explained Variance Ratio** for each component.
    - **SVD vs. Eig:** Implement PCA using Singular Value Decomposition (`np.linalg.svd`) and explain why SVD is numerically more stable for thin/tall matrices.

---

## Day 12: Nonlinear Projections (t-SNE & UMAP)
*Goal: Understanding Manifold Learning and Local vs. Global structure.*

- **Topics:**
    - **The Manifold Hypothesis:** High-dim data lies on low-dim manifolds.
    - **t-SNE:** Student’s t-distribution as a kernel to solve the "Crowding Problem."
    - **Perplexity:** How it acts as a "knob" for local vs. global attention.
- **Mastery Task:**
    - Use `scikit-learn` to visualize the MNIST dataset in 2D using PCA vs. t-SNE.
    - **Analysis:** Observe why PCA fails to separate digits that t-SNE clusters perfectly.
    - **UMAP:** Research why UMAP is faster and preserves more global structure than t-SNE.



---

## Day 13: SVM - The Primal Formulation
*Goal: Geometry of the Maximum Margin Classifier.*

- **Topics:**
    - **Functional vs. Geometric Margins.**
    - **The Optimization Objective:** Minimize $\frac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$.
    - **Hard Margin vs. Soft Margin:** Introducing Slack Variables ($\xi$) and the $C$ hyperparameter.
- **Mastery Task:**
    - Formalize the SVM Primal as a Quadratic Programming (QP) problem.
    - Use a library like `cvxopt` to solve the optimization for a small, linearly separable 2D dataset.

---

## Day 14: The Dual & The Kernel Trick
*Goal: Mastering Hilbert Spaces and Lagrangian Duality.*

- **Topics:**
    - **Lagrangian Duality:** Converting the Primal into the Dual form.
    - **The Kernel Trick:** Replacing the dot product $\langle x_i, x_j \rangle$ with a kernel function $K(x_i, x_j)$.
    - **Mercer’s Theorem:** Conditions for a function to be a valid kernel.
- **Mastery Task:**
    - Implement the **RBF (Gaussian) Kernel** and the **Polynomial Kernel**.
    - Derive why the Dual form is computationally efficient when feature dimensions $d$ are much larger than the number of samples $n$.



---

## Day 15: SVM Implementation & Hinge Loss
*Goal: Building a robust classifier from first principles.*

- **Topics:**
    - **Hinge Loss:** $L = \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$.
    - **Pegasos Algorithm:** Primal Estimated sub-GrAdient SOlver for SVM.
    - Relationship between SVM and Logistic Regression (Maximum Likelihood vs. Large Margin).
- **Mastery Task:**
    - Implement an SVM using **Stochastic Gradient Descent** on the Hinge Loss with $L_2$ regularization.
    - Compare your "SGD-SVM" accuracy and support vectors against the Scikit-Learn `SVC` (which uses LIBSVM/SMO).

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Math:** Linear Algebra (Eigendecomposition), Constrained Optimization (KKT Conditions).
- **Key Concept:** Dimensions are a curse, but the right "Basis" is a cure.