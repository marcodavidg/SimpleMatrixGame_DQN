import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import random
import matplotlib.pyplot as plt # Display graphs
import math
import os
# import memory
from environment import *
import deepq_network as dqn

tf.get_logger().setLevel('INFO')

pathTensorboard = "./tensorboard/dqn/1"
np.set_printoptions(linewidth=np.inf)

def pretrain(board: Environment, pretrain_length, memory: memory.Memory):
    state = board.get_state().copy()
    for i in range(pretrain_length):
        # Random action
        action = random.choice(possible_actions)

        # Do the action and get the new state and the reward that was obtained
        reward, new_state = board.move(action)
        if board.is_finished():
            # Add experience to memory
            memory.add([state, action, reward, new_state, True])
            board.reset()
            state = board.get_state().copy()
        else:
            memory.add((state, action, reward, new_state, False))
            # Our state is now the next_state
            state = new_state.copy()


def update_target_graph(DQNetwork, TargetNetwork):
    from_vars = DQNetwork.model.trainable_variables

    # Get the parameters of our TargetNetwork
    to_vars = TargetNetwork.model.trainable_variables

    op_holder = []

    # Update our target network parameters with DQNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

if __name__ == '__main__':

    # 1 up, 2 down, 3 left, 4 right
    possible_actions = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # MODEL HYPERPARAMETERS
    state_size = 2  # [coordinateX, coordinateY]
    lr = 0.001  # Alpha (the learning rate)

    # TRAINING HYPERPARAMETERS
    num_epochs = 1000  # Total episodes for training (N)
    max_steps = 100  # Max possible steps in an episode (L)
    batch_size = 20

    # FIXED Q TARGETS HYPERPARAMETERS
    max_tau = 128  # Tau is the C step where we update our target network

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # starting epsilon
    explore_stop = 0.1  # minimum exploration index, we should also try 0.01
    decay_rate = 0.001  # how fast the exploration index decreases. Also try 0.0001

    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate, the smaller the value, the less we care about future rewards. Let's also try 0.90

    # MEMORY HYPERPARAMETERS
    memory_size = batch_size * 31  # Number of experiences the memory can keep

    DeepQNetwork = dqn.DQN(state_size, possible_actions, "DeepQNetwork")
    TargetNetwork = dqn.DQN(state_size, possible_actions, "TargetNetwork")

    memory = memory.Memory(max_size=memory_size)

    board = Environment([4, 3])
    pretrain(board, batch_size, memory)
    board = Environment([4, 3])

    decay_step = 0  # Initialize the decay rate (that will use to reduce epsilon)
    tau = 0  # Set tau = 0

    for epoch in range(num_epochs):
        step = 0
        moves = 0
        episode_reward = 0
        board.reset()

        while step < max_steps:
            step += 1
            moves += 1
            decay_step += 1
            tau += 1

            # Forward pass to compute the loss
            # Q is our predicted Q value.
            output, explore_prob = DeepQNetwork.predict_action(explore_start, explore_stop, decay_rate, decay_step, board.get_state())

            # Do the action and get the new state and the reward that was obtained
            state = board.get_state().copy()
            reward, new_state = board.move(output)
            episode_reward += reward
            if board.is_finished():
                # board.draw()
                # Add experience to memory
                step = max_steps + 1
                memory.add([state, output, reward, new_state, True])
                board.reset()
                state = board.get_state().copy()
                print('Moves', moves, 'Episode:', epoch, 'Total reward:', round(episode_reward, 4),
                      'Won', board.wins, 'Lost', board.losses, "Epsilon:", round(explore_prob, 6))
            else:
                memory.add((state, output, reward, new_state, False))
                # Our state is now the next_state
                state = new_state.copy()

            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch])
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            ### DOUBLE DQN Logic
            # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
            # Use TargetNetwork to calculate the Q_val of Q(s',a')

            # Get Q values for next_state
            q_next_state = DeepQNetwork.model(next_states_mb)
            # Calculate Qtarget for all actions that state
            q_target_next_state = TargetNetwork.model(next_states_mb)

            for i in range(0, batch_size):
                last_move = dones_mb[i]

                # We got a'
                action = np.argmax(q_next_state[i])

                # If we are in a terminal state, only equals reward
                if last_move:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    # Take the Qtarget for action a'
                    target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                    target_Qs_batch.append(target)

            target_Qs = np.array([each for each in target_Qs_batch])

            loss = DeepQNetwork.calculate_loss(target_Qs, states_mb, actions_mb)

            # breakpoint()
            if tau > max_tau:
                # Update the parameters of our TargetNetwork with DQN_weights
                TargetNetwork.model.set_weights(DeepQNetwork.model.get_weights())

                tau = 0
                print("--------------------Target Network updated--------------------")

            if step == max_steps:
                print("max steps achieved")

        if epoch % 50 == 0:
            board.print_next()

        if epoch % 100 == 0:
            tf.keras.models.save_model(DeepQNetwork.model, 'saved_model.h5')
