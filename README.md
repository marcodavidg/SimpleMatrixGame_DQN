# Simple Matrix Game

This is a simple implementation of a Deep Q-Network (DQN) in TensorFlow 2 to solve a simple matrix console search game. This is a popular reinforcement learning algorithm for solving tasks in environments that provide discrete actions. 
In this README, we'll provide an overview of what the DQN is, and how the code works.

## About the project

A Deep Q-Network (DQN) is a type of neural network-based reinforcement learning algorithm that combines Q-learning with deep neural networks to learn how to make decisions in an environment. 
It's widely used in tasks like playing games and controlling robots. The key components of a DQN include:

- **Experience Replay**: Storing and sampling past experiences to break correlations and stabilize training.
- **Target Network**: Maintaining two copies of the network to stabilize learning.
- **Q-Learning Algorithm**: The Q-learning algorithm is used to update the Q-values based on the Bellman equation.

In this implementation, we have two identical models, often referred to as the "online" network and the "target" network. The purpose of these two networks is to improve the training stability of our reinforcement learning algorithm, specifically the Deep Q-Network (DQN).

**Online Network**: This is the primary network that interacts with the environment and learns from its experiences. It is updated and trained with every step in the environment.

**Target Network**: The target network is a copy of the online network, but its weights are updated less frequently. Instead of being trained with every step, the target network's weights are periodically copied from the online network.

The objective of having these two networks is to regulate the training process and improve the stability of learning. The target network plays a crucial role by providing target Q-values. These target Q-values serve as pseudo-labels for training the online network. By using these target Q-values, the online network can focus on improving its predictions based on a more stable and less frequently changing target, which results in smoother training and often faster convergence.


## Getting Started

1. **Environment Setup**:

   Before running the code, ensure you have Python and TensorFlow 2 installed on your system. You can install required packages using `pip`:

   ```bash
   pip install tensorflow
   

2. **Executing the code**:

   For executing the code we simply run the `main` file. If we want to just evaluate a previously saved model, we add the `--eval` and `--checkpoint` arguments:

   ```bash
   python main.py [--eval --checkpoint file_name]
   
# 
