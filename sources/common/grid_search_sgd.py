from grid_search import *
from sklearn.linear_model import SGDClassifier

def grid_search_sgd(name, features, labels, parameters, cv, checkpoint_path,
		scoring = None, checkpoint = True, single_estimator = True, checkpoint_step = 100, verbose = True, threads= -1, return_train_score = False):
	return grid_search(
		name=name,
		features=features,
		labels=labels,
		parameters=parameters,
		cv=cv,
		estimator_constructor=SGDClassifier,
		checkpoint_path=checkpoint_path,
		scoring=scoring,
		checkpoint=checkpoint,
		single_estimator=single_estimator,
		checkpoint_step=checkpoint_step,
		verbose=verbose,
		threads=threads,
		return_train_score=return_train_score
	)
