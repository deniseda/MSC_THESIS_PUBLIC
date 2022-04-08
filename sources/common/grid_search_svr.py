from grid_search import *
from sklearn.svm import SVR

def grid_search_svr(name, features, labels, parameters, cv, checkpoint_path, 
        checkpoint = True, single_estimator = True, checkpoint_step = 100, verbose = True, threads= -1):
        return grid_search(
            name = name,
            features = features,
            labels = labels,
            parameters = parameters,
            cv = cv,
            estimator_constructor= SVR,
            checkpoint_path = checkpoint_path,
            checkpoint = checkpoint, 
            single_estimator = single_estimator, 
            checkpoint_step = checkpoint_step, 
            verbose = verbose, 
            threads = threads, 
            scoring = None)


def generate_grid_search_features_labels_svr(injection_order, metabolites):
    injection_order_list = []
    metabolites_list = []
    for i in range(len(metabolites)):
        injection_order_list.append(injection_order.reshape(-1,1))
        metabolites_list.append(metabolites[i])
    return injection_order_list, metabolites_list



