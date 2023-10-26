import tensorflow as tf
import numpy as np
import random

# Define your custom loss function
def custom_loss(target_Qs, Qs):
    return tf.reduce_mean(tf.square(target_Qs - Qs))

class DQN():
    def __init__(self, state_size, possible_actions, name):
        # super(DQN, self).__init__()
        self.possible_actions = possible_actions
        self.action_size = len(possible_actions)
        self.name = name
        self.target_Q = None # target_Q is the R(s,a) + ymax Qhat(s', a')

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),  # Input layer with 4 features
            tf.keras.layers.Dense(128, activation='relu'),  # First hidden layer with 128 units and ReLU activation
            tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer with 64 units and ReLU activation
            tf.keras.layers.Dense(self.action_size, activation='softmax')  # Output layer with 10 units and softmax activation
        ])

        self.model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

    def calculate_loss(self, target_Qs, states_mb, actions_mb):
        with tf.GradientTape() as tape:
            Qs = tf.reduce_sum(tf.multiply(self.model(states_mb), actions_mb), axis=1)
            loss = custom_loss(target_Qs, Qs)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients to update model parameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")
        return loss


    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, is_pretrain=False, is_eval=False):
        # exploration - exploitation tradeoff
        # breakpoint()
        exp_exp_tradeoff = np.random.rand()
        possible_actions = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if not is_eval and (is_pretrain or explore_probability > exp_exp_tradeoff):
            # Choose a random action (exploration)
            action = random.choice(possible_actions)
            # print("RANDOM", action)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Qs = sess.run(DeepQNetwork.output, feed_dict={DeepQNetwork.inputs_: state.reshape((1, *state.shape))})
            # breakpoint()
            output = self.model(state[np.newaxis, :])
            # Take the biggest Q value (= the best action)
            choice = np.argmax(output)
            action = possible_actions[int(choice)]
            # print("PREDICTION", action)

        return action, explore_probability






