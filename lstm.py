import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras import layers, models
from keras import callbacks
import time

'''Global variables and parameters'''
GLOBAL_SEED = 27182818

'''Some initialisations and setup'''
np.random.seed(GLOBAL_SEED)


def row_to_2d(_row, _num_elems, _basic_cols, _special_cols):
    outer = []
    for i in range(_num_elems):
        inner = []
        for j in range(len(_basic_cols)):
            inner.append(_row[_basic_cols[j]])
        for j in range(len(_special_cols)):
            inner.append(_row[_special_cols[j] + '_' + str(i)])
        outer.append(inner)
    return outer


def set_to_list(_set):
    _special_cols = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']
    _basic_cols = _set.columns
    _num_elems = len([x for x in _basic_cols if _special_cols[0] in x])
    for i in range(len(_special_cols)):
        _basic_cols = [x for x in _basic_cols if _special_cols[i] not in x]

    lst = []
    for index, row in _set.iterrows():
        print('Set creation progress: ' + str(index) + '/' + str(len(_set)))
        lst.append(row_to_2d(row, _num_elems, _basic_cols, _special_cols))
    return lst


class LSTM:
    def __init__(self, training_set, training_labels, validation_set, validation_labels,
            optimiser='adam', epochs=200, batch_size=32, loss_func='mae', metrics=['mae'],
            activation_func='relu'
        ):
        self.training_set = training_set
        self.training_labels = training_labels 
        self.validation_set = validation_set
        self.validation_labels = validation_labels
        self.activation_func = activation_func
        self.optimiser = optimiser
        self.epochs = epochs 
        self.batch_size = batch_size
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
        self.model = models.Sequential([
            layers.LSTM(128, input_shape=(self.training_set.shape[1:]), return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.LSTM(128),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation=self.activation_func),
            layers.Dense(1)
        ])

    def train_model(self):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs // 20,
                                                 verbose=0, mode='auto')
        best_model = callbacks.ModelCheckpoint(filepath='models/lstm.model', monitor='val_loss', save_best_only=True)
        if self.model is not None:
            history = self.model.fit(
                self.training_set,
                self.training_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.validation_set, self.validation_labels),
                verbose=1, callbacks=[early_stopping, best_model]
            )
            self.model.load_weights('models/lstm.model')
            return history
        return None


'''Load data sets created by data_prep.py'''
# raw_training_set = pd.read_csv('data/training_set.csv')
raw_training_labels = pd.read_csv('data/training_labels.csv')
# raw_validation_set = pd.read_csv('data/validation_set.csv')
raw_validation_labels = pd.read_csv('data/validation_labels.csv')

'''Transform data sets into LSTM usable data sets'''
# training_set = set_to_list(raw_training_set)
# validation_set = set_to_list(raw_validation_set)
# print('Created lstm data sets')

'''Save the data sets'''
# with open('data/lstm_training_set.list', 'wb') as fp:
#    pickle.dump(training_set, fp)
# with open('data/lstm_validation_set.list', 'wb') as fp:
#    pickle.dump(validation_set, fp)
# print('Saved data')

'''Load data set previously created by the code above'''
print('Loading data')
with open("data/lstm_training_set.list", "rb") as fp:
    training_set = np.array(pickle.load(fp))
with open("data/lstm_validation_set.list", "rb") as fp:
    validation_set = np.array(pickle.load(fp))
print('Loaded data')

'''Create and train the LSTM'''
losses = []
val_losses = []
times = []
for i in range(5):
    print(str(i + 1), '/5')
    airqo_lstm = LSTM(training_set, raw_training_labels.to_numpy(), validation_set, raw_validation_labels.to_numpy())
    print('Created LSTM')
    tic = time.perf_counter()
    history = airqo_lstm.train_model()
    toc = time.perf_counter()
    elapsed = toc - tic
    times.append(elapsed)
    val_loss_min = min(history.history['val_loss'])
    index = history.history['val_loss'].index(val_loss_min)
    losses.append(history.history['loss'][index])
    val_losses.append(val_loss_min)
    print('Trained LSTM')
times = np.array(times)
losses = np.array(losses)
val_losses = np.array(val_losses)
print('Avg Training Time')
print(np.mean(times))
print('Avg Loss')
print(np.mean(losses))
print('Loss Stdev')
print(np.std(losses))
print('Avg Val Loss')
print(np.mean(val_losses))
print('Val Stdev')
print(np.std(val_losses))

'''Visually show the training done'''
train_label = airqo_lstm.metrics[0]
train_history = history.history[train_label]
validation_label = 'val_' + train_label
validation_history = history.history[validation_label]
plt.plot(train_history, label=train_label, linewidth=2, markersize=12)
plt.plot(validation_history, label=validation_label, linewidth=2, markersize=12)
plt.xlabel('Epoch')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.show()