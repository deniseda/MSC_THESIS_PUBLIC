import numpy as np
from matrix_precision import *

# rf_models --> random forests models previously saved
# metabolites --> QC or samples metabolites (peaks). 
# Depends on what you want predict


def get_prediction(rf_models, metabolites):
    prediction = []
    for i in range(len(rf_models)):
        prediction.append(rf_models[i].predict(metabolites.T))
    prediction = np.array(prediction, NP_METABOLITE)
    return prediction



def compute_median_value(metabolites):
    median_values = []
    for meta in metabolites:
        median_values.append(np.median(meta, axis=0))
    all_median_values = np.array(median_values)
    return all_median_values



def normalized_value(real_value, predict_value, median_value):
    final_values_normalized = []
    for i in range(0, len(real_value)):
        final_values_normalized.append((real_value[i] /predict_value[i]).T * median_value[i])
    normalized_values = np.array(final_values_normalized)
    return normalized_values