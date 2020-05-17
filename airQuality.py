import numpy as np
import pandas as pd

'''Global variables and parameters'''
GLOBAL_SEED = 27182818
VALIDATION_PROBABILITY = 0.2


# Split the data into training and validation sets
# Based on the VALIDATION_PROBABILITY
# Returns training set and validation set
def split_data(_training_set):
    _mask = np.random.rand(len(_training_set)) < (1 - VALIDATION_PROBABILITY)
    _train = _training_set[_mask]
    _validation = _training_set[~_mask]
    return _train, _validation


def normalise_data(_training_set, _validation_set, _testing_set):
    return _training_set, _validation_set, _testing_set


'''Some initialisations and setup'''
np.random.seed(GLOBAL_SEED)

'''Load training data'''
raw_data = pd.read_csv('Train.csv')

'''Load meta data'''
meta_data = pd.read_csv('airqo_metadata.csv')

'''Put default values in for NaN (based on minimum requirement for the NaN)'''
meta_data['dist_motorway'] = meta_data['dist_motorway'].fillna(5000)
meta_data['dist_trunk'] = meta_data['dist_trunk'].fillna(5000)
meta_data['dist_primary'] = meta_data['dist_primary'].fillna(5000)
meta_data['dist_secondary'] = meta_data['dist_secondary'].fillna(5000)
meta_data['dist_tertiary'] = meta_data['dist_tertiary'].fillna(5000)
meta_data['dist_unclassified'] = meta_data['dist_unclassified'].fillna(5000)
meta_data['dist_residential'] = meta_data['dist_residential'].fillna(5000)

training_set, validation_set = split_data(raw_data)
