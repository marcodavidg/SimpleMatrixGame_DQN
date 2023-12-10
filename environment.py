import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import random
import matplotlib.pyplot as plt # Display graphs
import math
import os
import pandas
import time

import deepq_network
import memory
class Environment:
    def __init__(self, size, init_pos, reward_pos, eval=False):
        self.wins = 0
        self.losses = 0
        self.points_lost = -1000
        self.points_reward = 10000
        self.size = np.array(size)
        self.reward_pos = np.array(reward_pos)
        self.board = self.get_new_board(eval)
        self.x_init, self.y_init = init_pos
        self.pos = np.array([self.x_init, self.y_init])
        self.printing = False

    def reset(self):
        self.board = self.get_new_board()
        while True:
            self.x_init = np.random.randint(1, self.size[0]-1)
            self.y_init = np.random.randint(1, self.size[1]-1)
            self.pos = np.array([self.x_init, self.y_init])
            if self.board[self.pos[0],self.pos[1]] != self.points_reward:
                break


    def get_new_board(self, eval):
        board = np.zeros((self.size[0] + 1, self.size[1] + 1))
        board[0, :] = self.points_lost
        board[-1, :] = self.points_lost
        board[:, 0] = self.points_lost
        board[:, -1] = self.points_lost
        if not eval:
            self.reward_pos = np.array([np.random.randint(1, self.size[0]-1), np.random.randint(1, self.size[1]-1)])
        board[self.reward_pos[0], self.reward_pos[1]] = self.points_reward
        return board

    def left(self):
        self.pos[1] -= 1

    def right(self):
        self.pos[1] += 1

    def down(self):
        self.pos[0] += 1

    def up(self):
        self.pos[0] -= 1

    def move(self, move):
        a = self.pos.copy()
        if move[0] == 1:
            self.pos[0] -= 1
        elif move[1] == 1:
            self.pos[0] += 1
        elif move[2] == 1:
            self.pos[1] -= 1
        elif move[3] == 1:
            self.pos[1] += 1
        reward = self.get_reward()
        # print("moved:", move, "Estaba en ", a[0], a[1], "y ahora en", self.pos[0], self.pos[1])
        if self.printing:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.draw()
            time.sleep(0.3)
        return reward, self.get_state()

    def draw(self):
        x = self.board.copy()
        x[self.pos[0], self.pos[1]] = 55
        print(x)

    def get_state(self):
        state = np.concatenate((self.pos, (self.reward_pos - self.pos)))
        return state

    def get_current_value(self):
        return self.board[self.pos[0], self.pos[1]]

    def get_reward(self):
        # Return Current reward - Euclidean distance to reward
        return self.get_current_value() - np.linalg.norm(self.pos - self.reward_pos)

    def is_finished(self):
        finished = (self.get_current_value() != 0)
        if finished:
            if self.get_current_value() == self.points_lost:
                self.losses += 1
                if self.printing:
                    print("YOU LOSE! :(")
            else:
                self.wins += 1
                if self.printing:
                    print("YOU WIN! :)")
            self.printing = False
        return finished

    def get_ratio(self):
        return (self.wins + 1) / (self.losses + 1)

    def print_next(self):
        self.printing = True
