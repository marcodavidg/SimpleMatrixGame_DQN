# Simple Matrix Game

This is a simple implementation of a Deep Q-Network (DQN) in TensorFlow 2 to solve a simple matrix console search game. This is a popular reinforcement learning algorithm for solving tasks in environments that provide discrete actions. 
In this README, we'll provide an overview of what the DQN is, and how the code works.

## What is a Deep Q-Network (DQN)?

A Deep Q-Network (DQN) is a type of neural network-based reinforcement learning algorithm that combines Q-learning with deep neural networks to learn how to make decisions in an environment. 
It's widely used in tasks like playing games and controlling robots. The key components of a DQN include:

- **Experience Replay**: Storing and sampling past experiences to break correlations and stabilize training.
- **Target Network**: Maintaining two copies of the network to stabilize learning.
- **Q-Learning Algorithm**: The Q-learning algorithm is used to update the Q-values based on the Bellman equation.

## Getting Started

1. **Environment Setup**:

   Before running the code, ensure you have Python and TensorFlow 2 installed on your system. You can install required packages using `pip`:

   ```bash
   pip install tensorflow
   

2. **Executing the code**:

   For executing the code we simply run the `main` file:

   ```bash
   python main.py
   
# 
