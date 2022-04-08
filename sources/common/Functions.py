from perseuspy import pd
from perseuspy.parameters import *
import numpy as np
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_error, balanced_accuracy_score, f1_score






def generate_features_labels(matrix, injection_order, qc_index):
    labels = matrix[qc_index] # takes ith mets
    submatrix = np.delete(matrix, qc_index, axis=0) # delete ith row
    features = np.vstack((injection_order, submatrix))
    return features.T, labels

#################################################################################################



def generate_gridsearch_features_labels(matrix, injection_order):
    features_list = []
    labels_list = []
    for i in range(0, len(matrix)):
        features, labels = generate_features_labels(matrix, injection_order, i)
        features_list.append(features)
        labels_list.append(labels)
    return features_list, labels_list


###################################################################################


def print_results(ytest, yprediction):
    scores = ''
    accuracy_score_value = "%.3f" % accuracy_score(ytest, yprediction)
    balanced_accuracy_score_value = '%.3f' % balanced_accuracy_score(ytest, yprediction)
    f1_score_value = '%.3f' % f1_score(ytest, yprediction, average='weighted')
    scores += 'Accuracy: ' + str(accuracy_score_value) + '\n' + 'Balanced accuracy: ' + str(balanced_accuracy_score_value) + '\n' + 'F1 score: ' + str(f1_score_value) + '\n'
    return scores



