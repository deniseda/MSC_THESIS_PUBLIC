import os
import sys
import pickle
import time as tm   
import timeit
from sklearn.model_selection import GridSearchCV




# global variables
model_name = ""
grids = []
time = []





def get_mean_elapsed_time(val):
    global time
    if len(time) != 10:
        time.append(val)
    else:
        time.pop(0)                 # remove first value
        time.append(val)
    return sum(time) / len(time)


def save_model(checkpoint_path):
    global grids
    global model_name
    pickle.dump( grids, open(checkpoint_path + "\\" + model_name , 'wb'))



# name = name of grid_search result dump
# estimator_constructor = constructor's function of the classifier or regressor
# features = X
# labels = y
# parameters = all estimator's parameters
# checkpoint_path = folder's path
# cv = cross validation type
# checkpoint = saves model progress while run, default to True
# single_estimator = default to True in this way creates one estimator --> obtaining only one gridsearch result
#   if single_estimator = False --> obtaining one estimator for each row 
# checkpoint_step interval between each checkpoint
# verbose = whether or not it should print the current status
# threads = number of CPUs used for gridsearch

def grid_search(name, features, labels, parameters, cv, estimator_constructor, checkpoint_path, scoring=None, checkpoint = True,
                 single_estimator = True, checkpoint_step = 100, verbose = True, threads = -1, return_train_score = False):
    global grids
    global time
    global model_name
    grids = []
    time = []
    model_name = name + '.gridres'
    if single_estimator:
        estimator = estimator_constructor()
        grid = GridSearchCV(estimator, parameters, cv = cv, scoring = scoring, n_jobs = threads, return_train_score = return_train_score)
        grid.fit(features, labels)
        grids.append(grid)
    else:                                                                                          
        if os.path.exists(checkpoint_path + '\\' + model_name):
            grids = pickle.load(open(checkpoint_path + '\\' + model_name, 'rb'))
        begin = len(grids)
        if checkpoint:
            if len(features) - begin < checkpoint_step:
                checkpoint_step = 10
        remaining_elems = checkpoint_step
        for n in range(begin, len(features)):
            if verbose:
                start_time = timeit.default_timer()
                print('-> Processing ' + str(n + 1) + ' of ' +  str(len(features)) +  ' elements ', end='\t')
                if checkpoint:
                    print('Checkpoint in ' +  str(remaining_elems) + ' elements', end='\t') 
            estimator = estimator_constructor()
            grid = GridSearchCV(estimator, parameters, cv = cv, scoring = scoring, n_jobs = threads, return_train_score= return_train_score)
            grid.fit(features[n], labels[n])
            grids.append(grid)
            if checkpoint:
                remaining_elems -=1 
            if verbose:
                stop_time = timeit.default_timer()
                remaining_time = get_mean_elapsed_time(stop_time - start_time) * (len(features) - n)
                remaining_time_string = tm.strftime('%H:%M:%S', tm.gmtime(remaining_time))
                print('Estimated time remaining ' + remaining_time_string, end='\n')
            if checkpoint and remaining_elems == 0:
                save_model(checkpoint_path)
                remaining_elems = checkpoint_step
    if checkpoint:
        save_model(checkpoint_path)
    if verbose:
        print('\nGrid search completed!')
    return grids

