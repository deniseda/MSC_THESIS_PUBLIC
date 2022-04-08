# This python file contains all common functions 
# used in all jupyter notebooks


from perseuspy import pd
from perseuspy.parameters import *
import numpy as np
import re
from matrix_precision import *







# ----------    COMMON FUNCTIONS -------------

def slice_submatrix(matrix, regex):
    return matrix.filter(regex=regex)


def slice_submatrix_by_labels(matrix, labels):
    return matrix.loc[:, labels]


def rename_header(header, regex):
    return re.sub(regex, '', header)


def rename_matrixheader_labels(matrix, regex):
    dictionary = {}
    new_columns = []
    for i in range(len(matrix.columns)):                    
        new_header = rename_header(matrix.columns[i][1], regex)
        new_columns.append(str(new_header))
        dictionary[matrix.columns[i][1]] = str(new_header)
    matrix.columns = new_columns
    return dictionary, matrix


def rename_matrixheader(matrix,regex):
    dictionary = {}
    new_columns = []
    for i in range(len(matrix.columns)):                    
        new_header = rename_header(matrix.columns[i][0], regex)
        new_columns.append(int(new_header))
        dictionary[matrix.columns[i][0]] = int(new_header)
    matrix.columns = new_columns
    return dictionary, matrix


def rename_specific_matrixheader(matrix, dictionary):
    new_columns = []
    for col in matrix.columns:
        new_columns.append(dictionary[col[0]])
    matrix.columns = new_columns
    matrix.columns = matrix.columns.astype(int)
    return matrix
    


def merge_submatrix(matrix, submatrix, regex):
    normalized_matrix = matrix.copy()
    matrix_rename_headers = matrix.copy()
    matrix_rename_headers = rename_matrixheader(matrix_rename_headers, regex)
    for i in range ( 0, len(submatrix.columns)):
        for j in range(0, len(matrix_rename_headers.columns)):
            if matrix_rename_headers.columns[j] == submatrix.columns[i]:
                normalized_matrix.iloc[:, j] = submatrix.iloc[:, i] 
    return normalized_matrix


def merge_submatrix_plugin(matrix, qc_submatrix, sample_submatrix, dictionary):
	normalized_matrix = matrix.copy()
	for i in range(0, len(normalized_matrix.columns)):
		if normalized_matrix.columns[i][0] in dictionary.keys():
			curr_injection_order = int(dictionary[normalized_matrix.columns[i][0]])
			if curr_injection_order in sample_submatrix.columns:
				normalized_matrix.iloc[:, i] = sample_submatrix[curr_injection_order]
			else:
				normalized_matrix.iloc[:, i] = qc_submatrix[curr_injection_order]
	return normalized_matrix


def sort_by_injection_order(matrix):
    return matrix.sort_index(axis=1)



def get_injection_order(matrix):
    injection_order = []
    n = 0
    for element in matrix.columns:
        injection_order.append(int(str(element[0])))
        n+=1
    injection_order_np = np.array(injection_order, dtype = NP_INJECTION)
    return injection_order_np



def get_injection_order_plugin(matrix):
    injection_order = matrix.columns
    injection_order_np = np.array(injection_order, dtype = NP_INJECTION)
    return injection_order_np


'''
def _cut_metabolites(matrix):
    metabolites = []
    for row_index in range(0, len(matrix.index)):
        metabolite = []
        for column_index in range(0, len(matrix.columns)):
            metabolite.append(matrix.iat[row_index, column_index])
            metabolite_np = np.array(metabolite, dtype = NP_METABOLITE)
        metabolites.append(metabolite_np)
    metabolites_np = np.array(metabolites, dtype = NP_METABOLITE)
    return metabolites_np
    '''


def cut_metabolites(matrix):
    return np.array(matrix, dtype=NP_METABOLITE)


def paste_metabolites(matrix, metabolites):
    matrix_copy = matrix.copy()
    for row_index in range(0,len(matrix_copy.index)):
        for column_index in range(0, len(matrix_copy.columns)):
            matrix_copy.iat[row_index, column_index] = metabolites[row_index][column_index] # array numpy no iat
    return matrix_copy

def filter_matrix(matr, column, value):
    return matr[matr[column] == value]


def relevant_metabolite(metabolitematrix, relevantmatrix, quantity= None):
    relevant_metabolite_matrix = []
    for i in range(0, len(metabolitematrix)):
        if i in relevantmatrix.index:
            relevant_metabolite_matrix.append(metabolitematrix[i])
    if quantity is not None:
        relevant_metabolite_matrix = relevant_metabolite_matrix[:quantity]
    return relevant_metabolite_matrix


def multi_to_single_header(matrix, keep):
    if keep < 0 or keep >= len(matrix.columns[0]):
        raise ValueError('Header must be kept between 0 and ' + str(len(matrix.columns[0])-1))
    columns = []
    for col in matrix.columns:
        columns.append(col[keep])
    single_header_matrix = matrix.copy()
    single_header_matrix.columns = columns
    return single_header_matrix


def cluster(matrix):
    return matrix.sort_values(by='Labels')


# this function converted one matrix in vector, take two matrices: one matrix with injection order and the other one with metabolites values.
# the function returns two object (vector)

def matrix_to_vector (matrix_injec_ord, matrix_values_meta):
    vector_meta = []
    vector_inj_ord = []
    for i in range(0, len(matrix_values_meta)):
        for j in range(0, len(matrix_values_meta[i])):
            vector_meta.append(matrix_values_meta[i][j])
            vector_inj_ord.append(matrix_injec_ord[j])
    return vector_meta, vector_inj_ord


def save_data(dataframe, path):
	dataframe.to_csv(path, sep='\t')