import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

'''Global variables and parameters'''
GLOBAL_SEED = 27182818

'''Some initialisations and setup'''
np.random.seed(GLOBAL_SEED)


# Creates a 2d array from a dataset row
# Simulates creation of an image with 1 channel color (i.e black and white img)
# Returns the 2d array
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

# Creates a list of 2d arrays from a dataset
# Creates a 2d array of each row and puts them into a list
# Returns the list of 2d arrays
def set_to_list(_set):

    _special_cols = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']
    _basic_cols = _set.columns
    _num_elems = len([x for x in _basic_cols if _special_cols[0] in x])
    for i in range(len(_special_cols)):
        _basic_cols = [x for x in _basic_cols if _special_cols[i] not in x]

    lst = []
    for index, row in _set.iterrows():
        print('Img creation progress: ' + str(index) + '/' + str(len(_set)))
        lst.append(row_to_2d(row, _num_elems, _basic_cols, _special_cols))
    return lst


'''Load data sets created by data_prep.py'''
raw_training_set = pd.read_csv('training_set.csv')
raw_training_labels = pd.read_csv('training_labels.csv')
raw_validation_set = pd.read_csv('validation_set.csv')
raw_validation_labels = pd.read_csv('validation_labels.csv')
print('Loaded data sets')

'''Transform data sets into CNN usable data sets'''
training_set = set_to_list(raw_training_set)
validation_set = set_to_list(raw_validation_set)
print('Created "image" data sets')