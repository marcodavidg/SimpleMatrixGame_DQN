import tensorflow as tf      # Deep Learning library
import sys
from environment import *
import argparse
import deepq_network as dqn
import pygame

tf.get_logger().setLevel('INFO')

pathTensorboard = "./tensorboard/dqn/1"
np.set_printoptions(linewidth=np.inf)

def update_target_graph(DQNetwork, TargetNetwork):
    from_vars = DQNetwork.model.trainable_variables

    # Get the parameters of our TargetNetwork
    to_vars = TargetNetwork.model.trainable_variables

    op_holder = []

    # Update our target network parameters with DQNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Function to set player position
def set_player_position(row, col):
    global player_x, player_y
    player_x, player_y = col * cell_size, row * cell_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for evaluation or training mode.")
    parser.add_argument("--eval", action="store_true", help="Enable just evaluation mode. Provide a checkpoint path.")
    parser.add_argument("--continue_training", action="store_true", help="Enable training but with checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--init", type=str, help="Initial position. Example: 12,3", default="1,1")
    parser.add_argument("--reward", type=str, help="Reward position. Example: 5,4", default="12,12")

    args = parser.parse_args()
    checkpoint = args.checkpoint
    eval = args.eval
    continue_training = args.continue_training
    if (eval or continue_training) and checkpoint is None:
        print("Error, checkpoint not provided.")
        exit()

    pos_args = args.init.split(',')
    reward_args = args.reward.split(',')
    pos_args = [int(pos_args[0]), int(pos_args[1])]
    reward_args = [int(reward_args[0]), int(reward_args[1])]




    # Initialize Pygame
    pygame.init()

    # Set up display
    cell_size = 50
    grid_size = 10
    width, height = cell_size * grid_size, cell_size * grid_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Player Movement in Grid")

    # Set up player
    player_size = cell_size
    player_x, player_y = 0, 0
    player_speed = cell_size

    # Set up colors
    red = (255, 0, 0)
    grid_color = (0, 0, 0)

    # Main game loop
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get the state of all keys
        keys = pygame.key.get_pressed()

        # Update player position based on keys
        if keys[pygame.K_LEFT] and player_x > 0:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < width - player_size:
            player_x += player_speed
        if keys[pygame.K_UP] and player_y > 0:
            # player_y -= player_speed
            set_player_position(1, 1)
        if keys[pygame.K_DOWN] and player_y < height - player_size:
            player_y += player_speed

        # Draw grid
        for row in range(grid_size):
            for col in range(grid_size):
                pygame.draw.rect(screen, grid_color, (col * cell_size, row * cell_size, cell_size, cell_size), 1)

        # Draw background
        screen.fill((255, 255, 255))

        # Draw player
        pygame.draw.rect(screen, red, (player_x, player_y, player_size, player_size))

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(10)  # Adjust the frame rate as needed




    exit()


    # 1 up, 2 down, 3 left, 4 right
    possible_actions = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # MODEL HYPERPARAMETERS
    state_size = 4  # [coordinateX, coordinateY, rewardX, rewardY]
    lr = 0.001  # Alpha (the learning rate)

    # TRAINING HYPERPARAMETERS
    num_epochs = 1000  # Total episodes for training (N)
    max_steps = 100  # Max possible steps in an episode (L)
    batch_size = 64

    # FIXED Q TARGETS HYPERPARAMETERS
    max_tau = 128  # Tau is the C step where we update our target network

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # starting epsilon
    explore_stop = 0.1  # minimum exploration index, we should also try 0.01
    decay_rate = 0.001  # how fast the exploration index decreases. Also try 0.0001

    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate, the smaller the value, the less we care about future rewards.

    # MEMORY HYPERPARAMETERS
    memory_size = 1024  # Number of experiences the memory can keep

    DeepQNetwork = dqn.DQN(state_size, possible_actions, "DeepQNetwork")
    TargetNetwork = dqn.DQN(state_size, possible_actions, "TargetNetwork")

    memory = memory.Memory(max_size=memory_size)

    size = [15,15]

    # pretrain(board, batch_size, memory)
    # board = Environment(size, reward_position)

    decay_step = 0  # Initialize the decay rate (that will use to reduce epsilon)
    tau = 0  # Set tau = 0


    if eval:
        board = Environment(size, pos_args, reward_args, True)
        # Load model
        DeepQNetwork.model.load_weights(checkpoint)
        board.print_next()
        # Play one episode
        while not board.is_finished():
            board.draw()
            output, _ = DeepQNetwork.predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                    board.get_state(), is_eval=True)
            board.move(output)

    else:
        board = Environment(size)
        is_pretrain = True
        if continue_training:
            explore_start = 0.5


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
                output, explore_prob = DeepQNetwork.predict_action(explore_start, explore_stop, decay_rate, decay_step, board.get_state(), is_pretrain)

                # Do the action and get the new state and the reward that was obtained
                state = board.get_state().copy()
                reward, new_state = board.move(output)
                episode_reward += reward
                if board.is_finished():
                    # Add experience to memory
                    step = max_steps + 1
                    memory.add([state, output, reward, new_state, True])
                    # board.reset()
                    # state = board.get_state().copy()
                    print(f'Started in({board.x_init},{board.y_init}) Reward in ({board.reward_pos[0]},{board.reward_pos[1]}) Moves {moves}, Episode: {epoch} Total reward: {round(episode_reward, 4)} '
                          f'Won {board.wins} Lost {board.losses} Epsilon: {round(explore_prob, 6)}')
                else:
                    memory.add((state, output, reward, new_state, False))
                    # Our state is now the next_state
                    state = new_state.copy()

                if is_pretrain:
                    if decay_step == batch_size:
                        is_pretrain = False
                        decay_step = 0
                        tau = 0
                        print("Pretrain finalizado")
                    continue


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

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    TargetNetwork.model.set_weights(DeepQNetwork.model.get_weights())

                    tau = 0
                    print("--------------------Target Network updated--------------------")

                if step == max_steps:
                    print("max steps achieved")

            if epoch % 50 == 0:
                board.print_next()

            if epoch % 50 == 0:
                DeepQNetwork.model.save_weights('saved_DQN.ckpt')
                print("--------------------Network saved--------------------")

