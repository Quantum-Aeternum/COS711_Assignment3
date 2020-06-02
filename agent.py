import numpy as np
from dqn import DQN

class Agent():
    def __init__(self, gamma, epsilon, dqn, mem_size=10000, epsilon_end=0.01, epsilon_decay=0.999):

        self.gamma = gamma
        self.epsilon = epsilon
        self.dqn = dqn
        self.batch_size = dqn.batch_size
        self.observation_dim = dqn.observation_dim
        self.num_actions = dqn.num_actions
        self.action_space = [i for in range(num_actions)]
        self.mem_size = mem_size
        self.mem_counter = 0
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.state_memory = np.zeros((self.mem_size, *observation_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *observation_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):    
        index = self.mem_counter % self.mem_size    
        self.mem_counter += 1
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            return self.dqn.choose_action(observation)
        else:
            return np.random.choice(self.action_space)

    def learn(self):
        ''' skip learning if not enough data has been gathered yet'''
        if self.mem_counter < self.batch_size:
            return
        