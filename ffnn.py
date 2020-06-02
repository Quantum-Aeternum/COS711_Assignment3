import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import callbacks

'''Global variables and parameters'''
GLOBAL_SEED = 27182818

'''Some initialisations and setup'''
np.random.seed(GLOBAL_SEED)


class FFNN:
    def __init__(self, training_set, training_labels, validation_set, validation_labels,
            optimiser='adam', epochs=200, batch_size=32, architecture_scaler=2, 
            activation_func='relu', loss_func='mae', metrics=['mae']
        ):
        self.training_set = training_set
        self.training_labels = training_labels 
        self.validation_set = validation_set
        self.validation_labels = validation_labels
        self.optimiser = optimiser
        self.epochs = epochs 
        self.batch_size = batch_size
        self.architecture_scaler = architecture_scaler 
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.metrics = metrics 

        self.model = None
        self.build_model()
        self.compile_model()

    def compile_model(self):
        self.model.compile(
            loss=self.loss_func,
            optimizer=self.optimiser,
            metrics=self.metrics
        )

    def build_model(self):
        input_len = len(self.training_set[0])
        self.model = keras.Sequential([
            layers.Dense(2584, activation=self.activation_func, input_shape=[input_len]),
            layers.Dense(987, activation=self.activation_func),
            layers.Dense(377, activation=self.activation_func),
            layers.Dense(144, activation=self.activation_func),
            layers.Dense(34, activation=self.activation_func),
            layers.Dense(1)
        ])

    def train_model(self):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs//20, verbose=0, mode='auto')
        best_model = callbacks.ModelCheckpoint(filepath='models/ffnn.model', monitor='val_loss', save_best_only=True)
        if self.model is not None:
            history = self.model.fit(
                self.training_set,
                self.training_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.validation_set, self.validation_labels),
                verbose=1, callbacks=[early_stopping, best_model]
            )
            self.model.load_weights('models/ffnn.model')
            return history
        return None

        
training_set = pd.read_csv('data/training_set.csv')
training_labels = pd.read_csv('data/training_labels.csv')
validation_set = pd.read_csv('data/validation_set.csv')
validation_labels = pd.read_csv('data/validation_labels.csv')
print('Loaded data sets')
        
training_set = training_set.to_numpy()
training_labels = training_labels.to_numpy()
validation_set = validation_set.to_numpy()
validation_labels = validation_labels.to_numpy()
print('Converted data sets')

airqo_ffnn = FFNN(training_set, training_labels, validation_set, validation_labels)
print('Created FFNN')
history = airqo_ffnn.train_model()
print('Trained FFNN')

train_label = airqo_ffnn.metrics[0]
train_history = history.history[train_label]
validation_label = 'val_' + airqo_ffnn.metrics[0]
validation_history = history.history[validation_label]
plt.plot(train_history, label=train_label, linewidth=2, markersize=12)
plt.plot(validation_history, label = validation_label, linewidth=2, markersize=12)
plt.xlabel('Epoch')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.show()

airqo_ffnn.model.load_weights('models/ffnn.model')