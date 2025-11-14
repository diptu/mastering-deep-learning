# What Is a Loss Function?

A **loss function** is a mathematical way to measure how good or bad a model’s predictions are compared to the actual results.  
It outputs a **single number** that represents how far off the predictions are — the **smaller the number, the better** the model is performing.  

Loss functions are essential in training machine learning models because they:

### 🔧 1. Guide Model Training  
During training, algorithms such as **Gradient Descent** use the loss function to adjust the model’s parameters.  
The goal is to reduce the loss value so the model can make more accurate predictions.

### 📊 2. Measure Performance  
Loss functions compute the difference between predicted and actual values.  
This helps in **evaluating how well the model is performing**.

### 🧠 3. Affect Learning Behavior  
Different loss functions emphasize different types of errors.  
Choosing the right one can influence **how the model learns** and what kinds of mistakes it prioritizes avoiding.

---

## Regression Loss Functions

### 1. Mean Squared Error (MSE) Loss

**Mean Squared Error (MSE) Loss** is one of the most widely used loss functions for **regression tasks**.  
It calculates the **average of the squared differences** between the predicted values and the actual values.  

MSE is simple to understand and **sensitive to outliers**, because squaring the errors can amplify large differences.

The formula for MSE is:

<!-- \[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\] -->

<!-- Where:  
- \(n\) = number of samples  
- \(y_i\) = actual value of the \(i\)-th sample  
- \(\hat{y}_i\) = predicted value of the \(i\)-th  -->

MSE = (1/n) * Σ (y_i - y_hat_i)^2

Where:  
- `n` = number of samples  
- `y_i` = actual value of the i-th sample  
- `y_hat_i` = predicted value of the i-th sample

### Reference

- [Loss Functions in Deep Learning - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/loss-functions-in-deep-learning/)
