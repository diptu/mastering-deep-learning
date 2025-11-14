# 🚀 Mastering Deep Learning
- 🎯 Hands-On Deep Learning: From Scratch to Advanced Models

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)  ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi)  ![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-Framework-FF6F00?logo=tensorflow)  ![Road to Mastery](https://img.shields.io/badge/Road_to-Mastery-success?logo=github)

---

## Table of Contents
- [Overview](#overview)
- [Why This Roadmap?](#why-this-roadmap)
- [Roadmap](#roadmap)
- [Repository Layout](#layout)
- [Progress Checklist](#progress-checklist)
- [Future Extensions](#future-extensions)
- [Setup Instructions](#setup-instructions)

  
## Overview

Hands-on journey to **mastering Deep Learning**:

- **Theory + Classic Papers**  
- **NumPy implementations from scratch**  
- **PyTorch & TensorFlow practice projects**  
- **Optional FastAPI deployment**

---

## Why This Roadmap?

This roadmap guides learners from **core neural network fundamentals** to **advanced deep learning topics**:

- **Theory-first:** Start with **NumPy from scratch** implementations.  
- **Framework transition:** Move to **PyTorch/TensorFlow** for scalability and real-world projects.  
- **Hands-on projects:** MNIST, CIFAR, ImageNet, text datasets, and more.  
- **Comprehensive coverage:** ANN → Computer Vision → RNN → NLP/LLM → GAN → RL → AI Agents.

---

## Layout

```
mastering_deep_learning/
│── pyproject.toml        # UV package manager config
│── README.md             # Overview & roadmap
│── LICENSE
│── requirements.txt
│── src/                  # Core modules
│   ├── ann/
│   ├── cv/
│   ├── rnn/
│   ├── nlp/
│   ├── gan/
│   └── rl/
│── projects/             # Practice projects
│   ├── mnist.py
│   ├── fashion_mnist.py
│   └── custom_dataset.py
│── notebooks/            # Optional weekly notebooks
│── data/                 # Datasets
│── papers/               # Classic papers per week
```

---

## Roadmap

| 📅 Week | Phase | Focus | Project / Dataset | Framework | Paper / Reference | Resources / Blogs |
|------|-------|-------|-----------------|----------|-----------------|-----------------|
| 1 | ![ANN](https://img.shields.io/badge/ANN-FF6F61?logo=brain) | Neurons, Activations, Loss, Forward Pass | MNIST | NumPy | [McCulloch & Pitts, 1943](https://www.cse.chalmers.se/~coquand/AUTOMATA/mcp.pdf) | [Lil's log](https://lilianweng.github.io/posts/2017-06-21-overview/), [Nielsen Book](http://neuralnetworksanddeeplearning.com/) |
| 2 | ![ANN](https://img.shields.io/badge/ANN-FF6F61?logo=brain) | Backpropagation & Gradient Descent | MNIST | NumPy, PyTorch | [Rumelhart et al., 1986](https://www.nature.com/articles/323533a0) | [Colah Backprop](http://colah.github.io/posts/2015-09-Backprop/), [Karpathy Recipe](http://karpathy.github.io/2019/04/25/recipe/) |
| 3 | ![ANN](https://img.shields.io/badge/ANN-FF6F61?logo=brain) | Optimizers, Regularization, BatchNorm | Fashion-MNIST | PyTorch, TensorFlow | [Dropout, 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) | [CS231n – Regularization](http://cs231n.stanford.edu/notes2021/notes2021.pdf) |
| 4 | ![ANN](https://img.shields.io/badge/ANN-FF6F61?logo=brain) | Hyperparameter Tuning, Debugging, Metrics | Fashion-MNIST + Custom | PyTorch, TensorFlow | [BatchNorm, 2015](https://arxiv.org/abs/1502.03167) | [BatchNorm Blog](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) |
| 5-12 | ![CV](https://img.shields.io/badge/Computer_Vision-007acc?logo=opencv) | Convolutions, Pooling, CNN Architecture, Transfer Learning, Object Detection (YOLO/PASCAL VOC) | MNIST, Fashion-MNIST, CIFAR-10, ImageNet | PyTorch, TensorFlow | [LeCun, 1998], [Krizhevsky, 2012], [Redmon, 2016] | [CS231n](http://cs231n.stanford.edu/), [YOLO Blog](https://pjreddie.com/darknet/yolo/) |
| 13-14 | ![RNN](https://img.shields.io/badge/Sequence_Modeling-7B68EE?logo=python) | RNN, LSTM, GRU, Backprop Through Time | Time Series / Text | PyTorch | [Hochreiter & Schmidhuber, 1997], [Mikolov, 2010] | [Colah LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), [Karpathy RNN Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
| 15-16 | ![NLP](https://img.shields.io/badge/NLP-FF4500?logo=huggingface) | Transformers, BERT, GPT / LLM Fine-tuning | IMDB / Custom | PyTorch, TensorFlow | [Vaswani, 2017], [Radford, 2019] | [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/), [Hugging Face Course](https://huggingface.co/course/chapter1) |
| 17-18 | ![GAN](https://img.shields.io/badge/GAN-800080?logo=tensorflow) | GAN Basics, DCGAN / Conditional GAN | MNIST / CIFAR-10 | PyTorch | [Goodfellow, 2014], [Radford, 2015] | [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) |
| 19-20 | ![RL](https://img.shields.io/badge/Reinforcement_Learning-32CD32?logo=python) | RL Basics – Q-Learning, Policy Gradients, DQN, Actor-Critic | OpenAI Gym / Atari | PyTorch | [Sutton & Barto, 2018], [Mnih, 2015] | [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp) |

---

## Progress Checklist

![ANN](https://img.shields.io/badge/ANN-Phase-FF6F61?logo=brain)  
- [ ] Weeks 1-4: ANN Fundamentals & NumPy / PyTorch / TensorFlow practice  

![CV](https://img.shields.io/badge/Computer_Vision-007ACC?logo=opencv)  
- [ ] Weeks 5-12: CNN, Transfer Learning, Object Detection, ImageNet / YOLO  

![RNN](https://img.shields.io/badge/Sequence_Modeling-7B68EE?logo=python)  
- [ ] Weeks 13-14: RNN / LSTM / GRU sequence modeling  

![NLP](https://img.shields.io/badge/NLP-FF4500?logo=huggingface)  
- [ ] Weeks 15-16: Transformers / LLM  

![GAN](https://img.shields.io/badge/GAN-800080?logo=tensorflow)  
- [ ] Weeks 17-18: GAN / DCGAN  

![RL](https://img.shields.io/badge/Reinforcement_Learning-32CD32?logo=python)  
- [ ] Weeks 19-20: Reinforcement Learning (Q-Learning, DQN, Actor-Critic)

---

## Future Extensions

- Advanced CNN architectures (ResNet, DenseNet, EfficientNet)  
- Transformer-based models & fine-tuning LLMs  
- GAN variants (StyleGAN, CycleGAN)  
- RL algorithms (PPO, A3C, Multi-agent systems)  
- AI Agents for decision-making & environment simulation  
- FastAPI + Docker deployment + CI/CD pipelines

---

## Setup Instructions

```bash
# Clone repo
git clone https://github.com/diptu/mastering-deep-learning.git
cd mastering-deep-learning

# Install with UV package manager
uv add

# OR install dependencies with pip
pip install -r requirements.txt
```
