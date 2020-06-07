import tensorflow as tf
import keras
from keras import layers, models
import numpy as np

# Deep Q-Network (Reinforcement learning technique)
class DQN():
    def __init__(self, observation_dim, num_actions, optimiser='adam', epochs=2, batch_size=32,
                activation_func='relu', loss_func='mae', metrics=['mae']):
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.optimiser = optimiser
        self.epochs = epochs 
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.metrics = metrics 

        self.model = None
        self.build_model()
        self.compile_model()

    def build_model(self):
        print([self.observation_dim, ])
        self.model = keras.Sequential([
            layers.Dense(128, activation=self.activation_func, input_shape=(self.observation_dim,)),
            layers.Dense(64, activation=self.activation_func),
            layers.Dense(self.num_actions)
        ])

    def compile_model(self):
        self.model.compile(
            loss=self.loss_func,
            optimizer=self.optimiser,
            metrics=self.metrics
        )

    def choose_action(self, observation):
        predictions = self.model.predict(observation)
        return predictions

    def train_one_step(self, observation, target):
        self.model.fit(
            observation,
            target,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(observation, target),
            verbose=0
        )

# test = DQN(observation_dim=5, num_actions=3)
# observation = np.array([1,1,1,1,1])
# print(observation.shape)
# print(test.choose_action(observation))