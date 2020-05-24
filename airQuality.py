import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

'''Global variables and parameters'''
GLOBAL_SEED = 27182818
VALIDATION_PROBABILITY = 0.2

# Dropping rows with any NaNs (i.e. 0.0) = 7212/15539 rows left
# Dropping rows with >= 80% NaNs (i.e. 0.8) = 15170/15539 rows left
MAX_NAN_PERC = 0.8

'''Some initialisations and setup'''
np.random.seed(GLOBAL_SEED)


# Removes NaN from row at _col_name by replacing NaN with average of row at _col_name
# Checks if enough non NaN data
# Returns NaN% < MAX_NAN_PERC
def remove_nan_col(_row, _col_name):
    _string_arr = (_row[_col_name]).split(',')
    _num_NaN = _string_arr.count('nan')
    if (_num_NaN == 0):
        return True
    if (_num_NaN / len(_string_arr)) > MAX_NAN_PERC:
        _row[_col_name] = _row[_col_name].replace('nan', '0')
        return False
    _float_arr = []
    for i in range(len(_string_arr)):
        if _string_arr[i] != 'nan':
            _float_arr.append(float(_string_arr[i]))
    avg = np.mean(_float_arr)
    _row[_col_name] = _row[_col_name].replace('nan', str(avg))
    avg = np.mean(_float_arr)
    return True


# Removes NaN from dataset by replacing NaN with averages
# If NaN is more than MAX_NAN_PERC of data, entire row is removed
# Returns set without NaNs
def remove_nan(_set):
    _cleaned_set = _set.copy()
    _col_names = ['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']
    for index, row in _cleaned_set.iterrows():
        for i in range(len(_col_names)):
            success = remove_nan_col(row, _col_names[i])
            if success:
                _cleaned_set.loc[index, _col_names[i]] = row[_col_names[i]]
            else:
                _cleaned_set.drop(_cleaned_set[_cleaned_set['ID'] == row['ID']].index, inplace=True)
                continue
    return _cleaned_set


# Split the data into training and validation sets as well as their label sets
# Based on the VALIDATION_PROBABILITY
# Returns training set, training labels, validation set, validation labels
def split_data(_set):
    _mask = np.random.rand(len(_set)) < (1 - VALIDATION_PROBABILITY)
    _train = _set[_mask]
    _validation = _set[~_mask]
    _train_labels = _train.pop('target')
    _validation_labels = _validation.pop('target')
    return _train, _train_labels, _validation, _validation_labels


# Normalises the training and validation sets (between 0 and 1)
# Based on the training data min and max statistics
# Returns normalised training set, normalised validation set
def normalise_data(_training_set, _validation_set):
    return _training_set, _validation_set


#'''Load training data'''
#raw_data = pd.read_csv('Train.csv')
#print('Loaded training data')

#'''Load meta data'''
#meta_data = pd.read_csv('airqo_metadata.csv', index_col=0)
#print('Loaded meta data')

#'''Put default values in for NaN (based on minimum requirement for the NaN)'''
#meta_data['dist_motorway'] = meta_data['dist_motorway'].fillna(5000)
#meta_data['dist_trunk'] = meta_data['dist_trunk'].fillna(5000)
#meta_data['dist_primary'] = meta_data['dist_primary'].fillna(5000)
#meta_data['dist_secondary'] = meta_data['dist_secondary'].fillna(5000)
#meta_data['dist_tertiary'] = meta_data['dist_tertiary'].fillna(5000)
#meta_data['dist_unclassified'] = meta_data['dist_unclassified'].fillna(5000)
#meta_data['dist_residential'] = meta_data['dist_residential'].fillna(5000)

#'''Remove NaN rows using statistical method'''
#no_nan_data = remove_nan(raw_data)
#print('Removed NaNs')

#'''Replace the data location with its meta data'''
#combined_data = pd.merge(no_nan_data, meta_data, on='location')
#combined_data.pop('location')
#print('Combined data with meta data')

#combined_data.to_csv(r'preprocessed_data.csv', index=False)

'''Load data set previously created by the code above'''
combined_data = pd.read_csv('preprocessed_data.csv')
print('Loaded preprocessed data')

training_set, training_labels, validation_set, validation_labels = split_data(combined_data)
print('Splitted data into training and validation sets')

training_set, validation_set = normalise_data(training_set, validation_set)
print('Normalised data sets')
