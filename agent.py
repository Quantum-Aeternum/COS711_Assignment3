import gym
import numpy as np
from dqn import DQN


class Agent:
    def __init__(self, gamma, epsilon, dqn, mem_size=10000, epsilon_end=0.01, epsilon_decay=0.999):

        self.gamma = gamma
        self.epsilon = epsilon
        self.dqn = dqn
        self.batch_size = dqn.batch_size
        self.observation_dim = dqn.observation_dim
        self.num_actions = dqn.num_actions
        self.action_space = [i for i in range(dqn.num_actions)]
        self.mem_size = mem_size
        self.mem_counter = 0
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.state_memory = np.zeros((self.mem_size, dqn.observation_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, dqn.observation_dim), dtype=np.float32)
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
        # skip learning if not enough data has been gathered yet
        # if self.mem_counter < self.batch_size:
        #     return

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.state_memory[batch]
        new_state_btach = self.new_state_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]
        action_batch = self.action_memory[batch]

        print(state_batch)
        print(new_state_btach)
        print(reward_batch)
        print(terminal_batch)
        print(action_batch)
        target = reward_batch + self.gamma * np.max(1)

        self.dqn.train_one_step(observation, target)
        self.epsilon *= self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end


env = gym.make('LunarLander-v2')
agent = Agent(0.99, 1.0, DQN(5, 3))
scores = []
epsilon_history = []
num_games = 10000

for i in range(num_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, new_observation, done)
        agent.learn()
        observation = new_observation
    scores.append(score)
    epsilon_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])
    print('Episode ', i, ': [score = %.2f' % score, '] [avg = %.2f' % avg_score, '] [epsilon = %.2f' % agent.epsilon,
          ']')
