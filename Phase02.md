## Phase 2: Deep Learning Foundations & CNNs (Days 1–15)

## Overview
This week transitions from "black box" library usage to mastering the mathematical mechanics of spatial hierarchies. You will build a Convolutional Neural Network from the ground up, focusing on tensor operations, backpropagation through spatial layers, and the optimization algorithms that power them.

---

## Day 1: The Convolutional Layer (The Feature Extractor)
*Goal: Implement the core mathematical operation of computer vision.*

- **Topics:**
    - **Cross-Correlation vs. Convolution:** Why deep learning libraries actually use cross-correlation.
    - **Kernels/Filters:** Feature detection (edges, textures) via $3 \times 3$ or $5 \times 5$ weight matrices.
    - **Hyperparameters:** Stride (step size) and Padding (preserving spatial resolution).
    - **Channel Depth:** Transitioning from RGB (3 channels) to deep feature maps.
- **Mastery Task:**
    - Implement a `convolve2d` function using `NumPy` with support for stride and padding.
    - **Experiment:** Apply a Sobel filter to an image to manually extract horizontal and vertical edges.
    


---

## Day 2: Nonlinearity & Pooling (Downsampling)
*Goal: Introduce translational invariance and control computational complexity.*

- **Topics:**
    - **ReLU (Rectified Linear Unit):** The sparsity effect and solving the vanishing gradient problem.
    - **Max Pooling vs. Average Pooling:** Spatial hierarchies and reducing sensitivity to small shifts.
    - **The Receptive Field:** Calculating how much of the original image a single neuron "sees."
- **Mastery Task:**
    - Implement a `MaxPooling` layer from scratch using window-based slicing.
    - Build a "Feed-Forward" block: Convolution $\rightarrow$ ReLU $\rightarrow$ MaxPool.
    


---

## Day 3: Backpropagation through Space (Spatial Gradients)
*Goal: Derive and implement the gradients for convolutional layers.*

- **Topics:**
    - **The Chain Rule for Tensors:** Flowing gradients back through the convolution operation.
    - **Weight Gradients:** $\frac{\partial L}{\partial W} = \text{Input} * \frac{\partial L}{\partial Y}$ (Convolution of input and output gradients).
    - **Dilation & Transposed Convolutions:** Understanding how to "upsample" gradients during the backward pass.
- **Mastery Task:**
    - Manually derive the gradient for a $2 \times 2$ filter.
    - Implement the `backward()` method for your Convolutional layer to update weights and pass gradients to the previous layer.

---

## Day 4: Full Architecture - Flattening to Softmax
*Goal: Bridge the gap between spatial features and categorical classification.*

- **Topics:**
    - **Flattening:** Transitioning from 3D feature volumes to 1D vectors.
    - **Fully Connected (Dense) Layers:** Interpreting the "Reasoning" phase of the CNN.
    - **Softmax & Cross-Entropy:** Converting raw scores (logits) into a probability distribution.
- **Mastery Task:**
    - Implement the `Softmax` function: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$.
    - Combine your spatial layers with a Dense layer to create a complete end-to-end CNN.
    - **Challenge:** Derive the gradient of Cross-Entropy Loss with respect to the Softmax inputs.

---

## Day 5: Optimization & Modern Regularization
*Goal: Ensuring convergence and preventing overfitting in deep networks.*

- **Topics:**
    - **Momentum & Adam:** Why adaptive learning rates are necessary for deep landscapes.
    - **Dropout:** Preventing co-adaptation by randomly zeroing out neurons during training.
    - **Batch Normalization:** Stabilizing internal covariate shift.
- **Mastery Task:**
    - Implement an **Adam Optimizer** from scratch using first and second moment estimates ($m_t$ and $v_t$).
    - **Visual Experiment:** Train your scratch CNN on the MNIST dataset and plot the "Accuracy vs. Epoch" curve.

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Libraries:** `NumPy`, `Matplotlib`.
- **Core Concept:** Avoid `PyTorch` or `TensorFlow` high-level APIs. Focus on vectorizing tensor operations to avoid nested loops (e.g., using `im2col` for efficient convolutions).

---
## Phase 1: CNN Evolution & Architectures (Days 6–10)

## Overview
This week bridges the gap between your "from scratch" NumPy foundations and industry-standard frameworks. You will implement the historic architectures that defined the deep learning revolution, moving from manual gradient calculation to leveraging Autograd and GPU acceleration.

---

## Day 6: The Pioneer—LeNet-5 from Scratch
*Goal: Solidify your NumPy framework by building the first successful CNN.*

- **Topics:**
    - **LeNet-5 Architecture:** Researching the specific connectivity of the C1, S2, C3, and S4 layers.
    - **Subsampling vs. Modern Pooling:** Understanding the original "Average Pooling + Trainable Coefficient" approach.
    - **Sigmoid vs. ReLU:** Observing why the original LeNet used Sigmoid/Tanh and the challenges of vanishing gradients.
- **Mastery Task:**
    - Using your Day 1–5 NumPy components, assemble the full **LeNet-5** pipeline.
    - **Constraint:** Successfully train it on MNIST to $>98\%$ accuracy without using a deep learning library.



---

## Day 7: Scaling Up—AlexNet via PyTorch
*Goal: Transition to PyTorch and understand the birth of Modern Deep Learning.*

- **Topics:**
    - **The AlexNet Breakthrough:** Use of ReLU, Dropout, and Data Augmentation.
    - **Multi-GPU Parallelism:** How AlexNet split filters across two GPUs (and how to replicate this sequentially).
    - **Local Response Normalization (LRN):** Why it was used then and why we use Batch Norm now.
- **Mastery Task:**
    - Implement the **AlexNet** class in PyTorch using `nn.Module`.
    - Load the CIFAR-10 dataset and implement a training loop using `torch.optim`.
    - **Analysis:** Compare the parameter count of AlexNet vs. LeNet-5.



---

## Day 8: Depth & Uniformity—VGG-16 and VGG-19
*Goal: Master the philosophy of "Small Kernels, Great Depth."*

- **Topics:**
    - **The $3 \times 3$ Rule:** Why stacking two $3 \times 3$ kernels is mathematically equivalent to one $5 \times 5$ kernel but with fewer parameters and more nonlinearity.
    - **VGG Configurations:** Understanding the modular blocks (Conv-Conv-Pool) that allow VGG to scale from 16 to 19 layers.
    - **Memory Bottlenecks:** Analyzing why VGG's fully connected layers consume the vast majority of its $138$M+ parameters.
- **Mastery Task:**
    - Build a "VGG-Block" function in PyTorch to programmatically generate **VGG-16** and **VGG-19**.
    - **Optimization:** Use `torch.cuda` to move your model to the GPU and measure the inference time difference.



---

## Day 9: Weight Initialization & Vanishing Gradients
*Goal: Solving the stability issues that arise when networks get deeper (like VGG).*

- **Topics:**
    - **Xavier (Glorot) vs. He Initialization:** Why the variance of weights must be scaled based on input/output units.
    - **Internal Covariate Shift:** How Batch Normalization accelerates training.
- **Mastery Task:**
    - Manually implement **He Initialization** for your VGG layers.
    - Conduct a "Sensitivity Test": Compare convergence speed between a model with zero-initialization vs. He-initialization.

---

## Day 10: Performance Profiling & Inference
*Goal: Evaluate the trade-offs between accuracy, parameters, and FLOPs.*

- **Topics:**
    - **FLOPs (Floating Point Operations):** Measuring computational complexity.
    - **Top-1 vs. Top-5 Error:** Standard metrics for ImageNet-scale models.
- **Mastery Task:**
    - Write a script to count the total trainable parameters in your LeNet, AlexNet, and VGG implementations.
    - **Visualization:** Create a bar chart comparing the parameter count vs. the accuracy achieved on a subset of ImageNet or CIFAR-100.

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Framework:** `PyTorch` (Core focus).
- **Hardware:** Access to a GPU (Google Colab/Kaggle) is highly recommended for VGG training.
- **Key Concept:** Architectural modularity—learning to think in "Blocks" rather than individual layers.


## Phase 3: Residual Learning & Deep Optimization (Days 11–15)

## Overview
This week addresses the "Degradation Problem" where deeper networks perform worse than shallower ones. You will master ResNet, the architecture that made training 100+ layer networks possible, and explore the advanced optimization techniques that define state-of-the-art computer vision.

---

## Day 11: The Residual Block (Identity Mapping)
*Goal: Understand the skip connection that revolutionized deep learning.*

- **Topics:**
    - **The Degradation Problem:** Why simply adding more layers increases training error.
    - **Residual Learning:** Learning the mapping $\mathcal{F}(x) = H(x) - x$ instead of the original $H(x)$.
    - **Identity Shortcuts:** How skip connections allow gradients to flow unimpeded through hundreds of layers.
- **Mastery Task:**
    - Implement a basic `ResidualBlock` in PyTorch.
    - **Challenge:** Handle the dimension mismatch when the stride is $> 1$ using $1 \times 1$ convolutions in the shortcut path.



---

## Day 12: ResNet-34 & ResNet-50 (Bottleneck Layers)
*Goal: Implement scalable architectures for complex datasets.*

- **Topics:**
    - **Basic vs. Bottleneck Blocks:** Why $1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ structures are used for deeper variants (ResNet-50, 101, 152).
    - **Global Average Pooling (GAP):** Replacing massive Fully Connected layers to reduce parameter count and prevent overfitting.
- **Mastery Task:**
    - Implement the **ResNet-50** architecture using the bottleneck pattern.
    - Compare the parameter count and training stability of ResNet-50 against VGG-19.



---

## Day 13: Advanced Backprop - Vanishing Gradients & Gradients Flow
*Goal: Mathematically verify why ResNet solves the gradient problem.*

- **Topics:**
    - **The Gradient Highway:** Deriving $\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} (1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i))$.
    - **Effective Depth:** Understanding how ResNet behaves like an ensemble of shallower networks.
- **Mastery Task:**
    - Use PyTorch hooks to visualize the gradient norms across layers in a standard CNN vs. a ResNet during training.
    - Observe the "gradient smoothing" effect provided by skip connections.

---

## Day 14: Regularization & Data Augmentation (Albumentations)
*Goal: Train deep models like ResNet on limited data without overfitting.*

- **Topics:**
    - **Advanced Augmentation:** Mixup, CutMix, and AutoAugment.
    - **Weight Decay vs. $L_2$:** The subtle differences in adaptive optimizers like AdamW.
    - **Label Smoothing:** Softening target distributions to improve generalization.
- **Mastery Task:**
    - Integrate the `Albumentations` library into your PyTorch DataLoader.
    - Implement a training script that utilizes **Mixup** augmentation and observe its impact on ResNet-18 validation accuracy.

---

## Day 15: Transfer Learning & Fine-Tuning
*Goal: Apply massive pre-trained models to niche downstream tasks.*

- **Topics:**
    - **Feature Extraction vs. Fine-Tuning:** Deciding which layers to freeze based on dataset similarity.
    - **Discriminative Learning Rates:** Using different $\eta$ for the backbone vs. the new head.
- **Mastery Task:**
    - Load a pre-trained **ResNet-50** (ImageNet weights).
    - Replace the final layer for a new task (e.g., Cat vs. Dog or Medical Imaging).
    - **Comparison:** Train the model with frozen vs. unfrozen weights and document the convergence speed.

---

## Technical Stack for the Week
- **Language:** Python 3.10+
- **Libraries:** `PyTorch`, `Torchvision`, `Albumentations`.
- **Key Concept:** **Residual Learning**—understanding that the easiest thing for a network to learn should be the identity function.