from grid_search import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from matrix_precision import *

# labels for binary classification

def create_label_dataframe(dataframe):
    labels = []
    header = dataframe.columns
    for label in header: 
        if label == 'control':
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)




def _create_feature_matrix(dataframe):
    features = []
    for i in range(0, len(dataframe.index)):
        features.append(dataframe.iloc[i].values)
    return np.array(features, dtype = NP_METABOLITE).T


# labels for multi classification

def create_multiclass_label_df(dataframe):
    labels = []
    header = dataframe.columns
    for label in header:
        if label == 'control':
            labels.append(0)
        elif label == 'local':
            labels.append(1)
        elif label == 'post':
            labels.append(2)
        else:
            labels.append(3)
    return np.array(labels)


# label for multi classification among sick categories

def create_multiclass_sick(df):
    labels = []
    header = df.columns
    for label in header:
        if label == 'local':
            labels.append(0)
        elif label == 'post':
            labels.append(1)
        elif label == 'onset':
            labels.append(2)
    return np.array(labels)


def create_feature_matrix_sick(df):
    sick_df = df.drop(labels='control', axis= 1)
    sick_df.reset_index()
    return np.array(sick_df).T



def grid_search_rf_classifier(name, features, labels, parameters, cv, checkpoint_path, 
        checkpoint = True, single_estimator = True, checkpoint_step = 100, verbose = True, threads= -1, return_train_score = True):
        return grid_search(name = name, 
        features = features, 
        labels = labels, 
        parameters = parameters, 
        cv = cv, 
        estimator_constructor =  RandomForestClassifier, 
        checkpoint_path = checkpoint_path, 
        checkpoint = checkpoint, 
        single_estimator = single_estimator, 
        checkpoint_step = checkpoint_step, 
        verbose = verbose, 
        threads = threads, 
        scoring = None, 
        return_train_score = return_train_score)


#################################################################################################################################################################################################

def generate_features_labels(matrix, injection_order, qc_index):
    labels = matrix[qc_index] # takes ith mets
    submatrix = np.delete(matrix, qc_index, axis=0) # delete ith row
    features = np.vstack((injection_order, submatrix))
    return features.T, labels


def generate_grid_search_features_labels_rf(injection_order, matrix):
    features_list = []
    labels_list = []
    for i in range(0, len(matrix)):
        features, labels = generate_features_labels(matrix, injection_order, i)
        features_list.append(features)
        labels_list.append(labels)
    return features_list, labels_list


def grid_search_rf_regressor(name, features, labels, parameters, cv, checkpoint_path, 
        checkpoint = True, single_estimator = True, checkpoint_step = 100, verbose = True, threads= -1, return_train_score = False):
        return grid_search(name = name, 
        features = features, 
        labels = labels, 
        parameters = parameters, 
        cv = cv, 
        estimator_constructor = RandomForestRegressor, 
        checkpoint_path = checkpoint_path, 
        checkpoint = checkpoint, 
        single_estimator = single_estimator, 
        checkpoint_step = single_estimator, 
        verbose = verbose, 
        threads = threads)








