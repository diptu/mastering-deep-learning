# Phase 1: *Neural Networks and Deep Learning* (NNDL) (Days 1–20)

## Overview
This week transitions from the "black box" usage of libraries to mastering the mathematical mechanics of linear models and the optimization algorithms that power them.


---

# Phase 1: The Neural Mechanics (Days 1–10)

## Day 1–2: Perceptrons to Sigmoid Neurons (NNDL Ch. 1)
*Goal: Understand the transition from step-functions to differentiable activation.*

- **Topics:**
    - **The Perceptron:** Evidence of NAND-gate completeness; why small weight changes can cause catastrophic output flips.
    - **The Sigmoid Identity:** $\sigma(z) \equiv 1/(1+e^{-z})$. Understanding why smoothness is the prerequisite for learning.
    - **Architecture Foundations:** Input layers, hidden layers, and the distinction between MLP and Recurrent networks.
- **Mastery Task:**
    - Build a `Perceptron` that can simulate an `OR` and `NAND` gate.
    - Implement a `SigmoidNeuron` and visualize how the output changes as you perturb $w$ and $b$ by a tiny $\Delta$.
    - **Math:** Derive the derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

---

## Day 3–4: The Calculus of Learning (NNDL Ch. 2)
*Goal: Mastery of the Backpropagation algorithm—the "Engine" of AI.*

- **Topics:**
    - **The Four Fundamental Equations:** $\delta^L$ (output error), $\delta^l$ (layer error), $\frac{\partial C}{\partial b}$, and $\frac{\partial C}{\partial w}$.
    - **The Hadamard Product:** Efficient vectorization of weight updates.
    - **Computational Graphs:** Tracking how a local error flows backward through the chain rule.
- **Mastery Task:**
    - **Whiteboard Session:** Manually derive the error $\delta^l$ for a 3-layer network using only the chain rule.
    - Implement the `backprop` method in a `Network` class using NumPy.
    - 

---

## Day 5–6: Improving the Way Neural Networks Learn (NNDL Ch. 3)
*Goal: Solving the learning slowdown and overfitting.*

- **Topics:**
    - **Cross-Entropy Loss:** Why it eliminates the "learning slowdown" caused by $\sigma'(z)$ in the Quadratic Cost.
    - **Softmax & Log-Likelihood:** Understanding the probability interpretation of output layers.
    - **Regularization Strategy:** $L_1, L_2$, and the intuition behind **Dropout** (preventing co-adaptation).
    - **Weight Initialization:** Why $N(0, 1)$ leads to neuron saturation and the logic of $1/\sqrt{n_{in}}$ scaling.
- **Mastery Task:**
    - Implement "Stochastic Gradient Descent" with **L2 Regularization**.
    - Compare the learning curves of Quadratic Cost vs. Cross-Entropy on the MNIST dataset.
    - 

---

## Day 7–8: The "Deep" in Deep Learning (NNDL Ch. 4 & 5)
*Goal: Visualizing Universality and the Vanishing Gradient Problem.*

- **Topics:**
    - **The Universality Theorem:** Visual proof that a single hidden layer can approximate any continuous function.
    - **The Vanishing Gradient:** Why gradients get exponentially smaller in earlier layers.
    - **Unstable Gradients:** How the product of weight matrices $W$ and derivatives $\sigma'$ determines the flow of information.
- **Mastery Task:**
    - **Visual Experiment:** Build a network that approximates a complex "step-like" function using only sigmoid neurons.
    - Track the magnitude of $\nabla b$ across layers during training; plot the "Gradient Magnitude vs. Layer Depth" to witness the vanishing gradient.

---

## Day 9–10: Convolutional Neural Networks (NNDL Ch. 6)
*Goal: Exploiting spatial structure and local patterns.*

- **Topics:**
    - **Local Receptive Fields:** The move from dense connectivity to local feature detection.
    - **Shared Weights & Biases:** Why translation invariance reduces parameter count.
    - **Pooling Layers:** Max-pooling vs. Average-pooling; achieving robustness to small spatial distortions.
- **Mastery Task:**
    - Implement a 2D Convolution operation from scratch using NumPy (sliding window).
    - Build a simple CNN (Conv -> Pool -> Fully Connected) and train it on MNIST.
    - **Bonus:** Visualize the "Filters" (kernels) after training to see the edge detectors.
    - 

---

# Phase 1: Neural Network from scrach (Days 11–15)