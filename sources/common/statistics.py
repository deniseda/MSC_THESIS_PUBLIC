from matplotlib.cbook import boxplot_stats
import scipy.stats
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from perseuspy import pd


def get_variance_mean(metabolites):
    variance = []
    mean =[]
    for met in range(len(metabolites)):
        variance.append(np.var(metabolites[met]))
        mean.append(np.mean(metabolites[met]))
    return np.array(variance), np.array(mean)



### for each model in QC (raw and predicted)
def get_r2_score(original, predicted):
	r2_scores = []
	for i in range(0, len(original)):
		r2_scores.append(r2_score(original[i], predicted[i]))
	return np.array(r2_scores)


### for each model in QC (raw and predicted)
def get_mse_score(original, predicted):
	mse_score = []
	for i in range(0, len(original)):
		mse_score.append(mean_squared_error(original[i], predicted[i]))
	return np.array(mse_score)

def get_MedAE(original, predicted):
	mae = []
	for i in range(0, len(original)):
		mae.append(median_absolute_error(original[i], predicted[i]))
	return np.array(mae)





# compare boxplot
# we want compare median and Q1 Q3 values of each metabolites
def get_percentile(labels, data):
	percentile = []
	for d in range(len(labels)):
		dictionary = {} # declare a dictionary
		dictionary['Q1'] = np.percentile(data[d], 25)
		dictionary['median'] = np.median(data[d]) 	
		dictionary['Q3'] = np.percentile(data[d], 75)
		dictionary['min'] = np.min(data[d])
		dictionary['max'] = np.max(data[d])
		percentile.append(dictionary)
	return pd.DataFrame(percentile)




def create_df_boxplot_stat(metab):
	metab = np.array(metab)
	return pd.DataFrame(boxplot_stats(np.array(metab)))

'''
def save_data(dataframe, path):
	dataframe.to_csv(path, sep='\t')'''


#### better lambda function
def count_r2_score(r2score, threshold):
	major_r2 = []
	for i in range(len(r2score)):
		if r2score[i] >= threshold:
			major_r2.append(r2score[i])
	return np.array(major_r2)
