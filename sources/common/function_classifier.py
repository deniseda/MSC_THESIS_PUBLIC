# This file contains common functions for SDGclassifier

from perseuspy import pd
from perseuspy.parameters import *
import numpy as np
import re
from sklearn.metrics import  f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matrix import *
from matrix_precision import *
from sklearn import tree 
import matplotlib.pyplot as plt
from collections import Counter



#-------- FUNCTIONS -----------


# NEW METHOD
# function that takes metabolites as features
# use drop when it is multi classification among sick categories

def create_feature_matrix(dataframe, drop=None):
	dataframe_copy = dataframe.copy()
	if drop is not None:
		dataframe_copy.drop(labels=drop, axis=1, inplace=True)
		dataframe_copy.reset_index()
	return cut_metabolites(dataframe_copy).T

###############################################################


# NEW METHOD
# label_1 is healthy --> class control 
# label_2 is sick  --> class all sick[local,post,onset]

def create_labels_binary(dataframe, label_1, label_2, class_1_list, class_2_list, discard=None):
	labels = []
	for label in dataframe.columns:
		if label in class_1_list:
			labels.append(label_1)
		elif label in class_2_list:
			labels.append(label_2)
		elif label in discard:
			continue
		else:
			raise Exception('Cannot divide the dataframe into two classes')
	return np.array(labels)




# drop the column control for multi classification among sick categories

def create_labels(dataframe, drop=None, index=None):
	dataframe_copy = dataframe.copy()
	if drop is not None:
		dataframe_copy.drop(labels=drop, axis=1, inplace=True)
		dataframe_copy.reset_index()
	if index is not None:
		return dataframe_copy.columns.get_level_values(index)
	else:
		return dataframe_copy.columns

#######################################################################################################


##### Function to create dataframe (plotting results)
#datalist is list of balanced score
#labellist is list of classifiers

def create_dataframe (labelslist, datalist):
	df = pd.DataFrame()
	for i in range(len(labelslist)):
		df[labelslist[i]]= datalist[i]
	return df





############################################################################################
##############################################################################################
# function relevant feature RF classifier
# this two functions give back all features even those with score 0


def relevant_feature_importances(classifier):
	saved = []
	for i in classifier.feature_importances_:
		saved.append(i)
	return np.array(saved)


def relevant_coefficients(classifier):
	saved = []
	for i in classifier.coef_:
		saved.append(i)
	return np.array(saved)


###############################################################################
# function to create dictionary that contains all feature_importances even those with score 0

def create_dict_feature_importances(matrix, feat_importances, metab_key):
	dictionary = {}
	for i in range(len(feat_importances)):
		dictionary[matrix.at[i,metab_key]] = feat_importances[i]
	return dictionary


def create_dict_coefficients(matrix, coef, metab_key, class_index = 0):
	dictionary = {}
	for i in range(len(coef[class_index])):
		dictionary[matrix.at[i, metab_key]] = coef[class_index][i]
	return dictionary


# dictionary contain 1 if feature is important or 0 if isn't

def from_dic_FI_todict_number(feat_dict, threshold_vls, greater_than_strict = True):
	dictio = {}
	for k in feat_dict.keys():
		values = feat_dict[k]
		count = 0
		if greater_than_strict:
			if values > threshold_vls:
				count += 1
		else:
			if values >= threshold_vls:
				count += 1
		dictio[k] = count
	return dictio        

# dictionary contains only features has values >= threshold in JSON file 
def filter_feature_importances_by_threshold(feat_dict, threshold_vls, greater_than_strict = True):
	dictio = {}
	for k in feat_dict.keys():
		values = feat_dict[k]
		if greater_than_strict:
			if values > threshold_vls:
				dictio[k] = values
		else:
			if values >= threshold_vls:
				dictio[k] = values
	return dictio

#dictionary composed by ID met as key and value 1. 
# that means this metabolite was considered like feature importance
#to save only keys with value = 1

def filter_feature_importances(dictionary):
	return dict(filter(lambda x: x[1] == 1, dictionary.items()))


def remove_zero_coefficients(coef_dictionary):
	dictio = {}
	for k in coef_dictionary.keys():
		if coef_dictionary[k] != 0:
			dictio[k] = coef_dictionary[k]
	return dictio



def  count_features_indictionary(dictionary):
	count = 0
	for i in enumerate(dictionary):
		count += 1
	return count

####################################################################################


def merge_dictionaries(dict1, dict2):
	return dict(Counter(dict1) + Counter(dict2))


# function to search in different df the same ID (of feature_importances)
def search_common_feature_importance(list_dict):
	keys_list = []
	for dictionary in list_dict:
		keys_list.append(dictionary.keys())
	common_elements = list(set.intersection(*map(set, keys_list)))
	return common_elements    



def get_metabolites_name_byID(matrix, ID_list, IDcolumn_label, name_column_label):
	metabolites = []
	for i in range(len(ID_list)):
		row = matrix[matrix[IDcolumn_label] == str(ID_list[i])]
		metabolites.append(row[name_column_label].values[0])
	return metabolites
	
###########################################################################################################


def create_dataframe_fromgrid_result(grid_data, list_param):
	df_gridata = pd.DataFrame(grid_data.cv_results_)
	df = pd.DataFrame()
	df = df_gridata[list_param]
	return df

######################################################################################

def plot_rf_trees(title, estimators, rows, columns, xsize, ysize, classes, fig_save= False, fig_path=None):
	fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(xsize, ysize))
	plt.suptitle(title)
	n = 0
	for i in range(0, rows):
		for j in range(0, columns):
			tree.plot_tree(estimators[n], ax = axes[i][j], filled= True, class_names= classes)
			n += 1
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()



def get_best_value(classifiers):
	values = ''
	for classifier in classifiers:
		parameters = classifier.get_params()
		for key in parameters:
			values += str(key) + ': ' +  str(parameters[key]) + '\n'
		values += '\n'
	return values