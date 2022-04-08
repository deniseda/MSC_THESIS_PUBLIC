
from cProfile import label
from turtle import color
from sklearn.metrics import ConfusionMatrixDisplay, r2_score
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from matrix import *








def display_performances_wl(list_df, estimators, title, xlabel, ylabel, xsize, ysize, marker, size, legend_list, fig_save = False, fig_path =None):
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	plt.xlabel(xlabel)
	if type(estimators[0]) == str:
		plt.xticks(range(len(estimators)), estimators)
	else:
		plt.xticks(estimators)
	plt.ylabel(ylabel)
	for i in range(len(list_df)):
		sb.scatterplot(data= list_df[i], x= list_df[i]['X'], y= list_df[i]['Y'], marker = marker[i], s = size)
	plt.legend(legend_list, loc='best')
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()

##### plot confusion matrix 

def function_subplots_confusionmatrix(list_matrix, rows, columns, list_title,subtitle, dimensionX, dimensionY, labels, fig_save=False, fig_path=None):
	fig, axes = plt.subplots(rows,columns , figsize=(dimensionX,dimensionY))
	fig.suptitle(subtitle)
	i = 0
	j = 0
	for n in range(len(list_matrix)):
		m = ConfusionMatrixDisplay(list_matrix[n], display_labels= labels[n])
		if (rows > 1):
			m.plot(ax = axes[i][j])
			axes[i][j].set_title(list_title[n])
		else:
			m.plot(ax = axes[j])
			axes[j].set_title(list_title[n])
		j += 1
		if j == columns:
			i +=1
			j = 0
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()


######################################################

def function_subplots_testscore_gridsearch(dataframe, maxfeatures_labels, scores, list_title, fig_save= False, fig_path = None):
	fig, axes = plt.subplots(2, figsize=(20,15))
	for i in range(len(maxfeatures_labels)):
		newdf = dataframe[(dataframe['param_max_features'] == maxfeatures_labels[i])]
		for j in range(len(scores)):
			sb.lineplot(ax=axes[i], data= newdf, x = newdf['param_max_depth'], y=scores[j])
			axes[i].set_ylabel('score')
			axes[i].set_xlabel('max_depth')
			axes[i].set_xticks([x for x in newdf['param_max_depth']])
			axes[i].set_title(list_title[i])
			axes[i].legend(scores, loc='best')
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()



############################################
# raw-log -comparison 

def plot_distribution(dataframe, title,kde = True, ylim = None):
	plt.figure(figsize=(10,10))
	plt.title(title)
	sb.distplot(dataframe, kde = kde)
	if ylim != None:
		plt.ylim(ylim)




############################################################################################################
## SVR

def plot_metabolite(x_qc, y_qc, x_sample, y_sample,  xsize, ysize, title, color_qc = 'red', color_sample = 'black', marker_qc= 'X', marker_sample = 'H' ):
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	sb.scatterplot(x = x_qc, y = y_qc, color = color_qc, marker = marker_qc)
	sb.scatterplot(x = x_sample, y= y_sample, color = color_sample, marker= marker_sample)
	plt.legend(['QC', 'Samples'], loc = 'best')


# plot of variance(statistical analysis)

def plot_variance(title, var_raw, var_norm, xsize, ysize, label1, label2, alpha_q = .6, alpha_n =.2, color_r = 'black',
					bin = 50, color_n ='red' ,stat= 'count', save_fig = False, fig_path = None ):
	plt.figure(figsize=(xsize, ysize))
	sb.histplot(var_raw, color = color_r, bins = bin, stat= stat, alpha= alpha_q, label = label1)
	sb.histplot(var_norm, color = color_n, bins = bin, stat = stat, alpha = alpha_n, label = label2)
	plt.title(title)
	plt.legend()
	if save_fig:
		plt.savefig(fig_path, format = 'svg')
	plt.show()



#################

def variance_plot(data, bins, xsize, ysize, title, stat = 'count', color = 'red',  fig_save=False, fig_path ='svg'):
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	sb.histplot(data = data, bins = bins, stat= stat, color= color)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()




#like histogram
def plot_r2_distribution(r2_score, xsize, ysize, title, color='red', fig_save= False, fig_path = None):
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	sb.histplot(r2_score, color=color)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()



def plot_scatterR2(title, original_data, predicted_data, xlabel, ylabel, xsize, ysize, marker = '*', color = 'red', edgecolors = 'black', alpha= .2, fig_save= False, fig_path = None):
	plt.figure(figsize=(xsize,ysize))
	plt.title(title + ' R2: ' + str(r2_score(original_data, predicted_data)))
	plt.scatter(original_data, predicted_data, marker = marker, color = color, edgecolors= edgecolors, alpha= alpha)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()




def plot_filter_meta_boxplot(x_InjOrd, y_meta, xsize, ysize, title, color = 'black', showmeans = True, showfliers= True, fig_save=False, fig_path=None):
	plt.figure(figsize=(xsize, ysize))
	plot = sb.boxplot(x=x_InjOrd,y= y_meta, showmeans = showmeans, showfliers = showfliers, flierprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 10})
	plot = sb.stripplot(x=x_InjOrd, y=y_meta, color=color)
	plot.set_xticklabels(labels = [item.get_text() for item in plot.get_xticklabels()], rotation=60)
	plot.set_ylabel('intensity')
	plot.set_xlabel('injection order')
	plot.set_title(title)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()


# function for facet_plot


def generate_data_facet_plot(injection_order, relevant_metabolites):
	if len(relevant_metabolites) > 50:
		print("You can't plot more than 50 metabolites")
		return
	number = []
	group = []
	for i in range(0, len(relevant_metabolites)):
		for metabolite in relevant_metabolites[i]:
			number.append(i + 1)
	for i in range(0, len(relevant_metabolites)):
		if i >= 0 and i < 10:
			for j in range(0, len(relevant_metabolites[i])):
				group.append("Between 1 and 10")
		elif i >= 10 and i < 20:
			for j in range(0, len(relevant_metabolites[i])):
				group.append("Between 11 and 20")
		elif i >= 20 and i < 30:
			for j in range(0, len(relevant_metabolites[i])):
				group.append("Between 21 and 30")
		elif i >= 30 and i < 40:
			for j in range(0, len(relevant_metabolites[i])):
				group.append("Between 31 and 40")
		elif i >= 40 and i < 50:
			for j in range(0, len(relevant_metabolites[i])):
				group.append("Between 41 and 50")
	metabolites, order = matrix_to_vector(injection_order, relevant_metabolites)
	data = pd.DataFrame()
	data["Number"] = number
	data["Intensity"] = metabolites
	data["Group"] = group
	return data

def create_facetgrid_plot(dataframe, xsize, ysize, title, fig_save = False, fig_path=None):
	facetplot = sb.FacetGrid(dataframe, col = 'Group', sharex= False)
	facetplot.fig.suptitle(title)
	facetplot.map_dataframe(sb.boxplot, 'Number', 'Intensity', showmeans= True)
	facetplot.map_dataframe(sb.stripplot, 'Number', 'Intensity', palette= 'hls', dodge= False)
	facetplot.fig.set_figwidth(xsize)
	facetplot.fig.set_figheight(ysize)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()



def create_pvalue_plot(f_value, p_value, metabolites, xsize, ysize):
	plt.figure(figsize=(xsize, ysize))
	plt.title('Plot of F-test results')
	pvalue_plot = sb.lineplot(x = range(1, len(metabolites) + 1), y = f_value, markers= True, marker = '*', color = 'red')
	pvalue_plot = sb.lineplot(x = range(1, len(metabolites) + 1), y = p_value, markers= True, marker = 'o', color = 'black')
	pvalue_plot.set_xticks(range(1, len(metabolites) + 1), rotation=60)
	pvalue_plot.set_ylabel('values')
	pvalue_plot.set_xlabel('Number of metabolites')
	pvalue_plot.set_ylim(-1, 5, 0.5)
	plt.legend(['f_value', 'p-value' ], loc = 'best')
	plt.show()



def subplots_filtered_meta_nodrift(title, qc_injection_order, qc_filter_pred_met, sample_injection_order, sample_filter_pred_meta, xlim, ylim, xsize, ysize,nrows = 10, ncols = 5 , alpha_s = .1, alpha_q = .9, fig_save = False, fig_path = None):
	fig, axes = plt.subplots(nrows= nrows, ncols= ncols, figsize=(xsize, ysize))
	plt.suptitle(title)
	n = 0
	for i in range(0, nrows):
		for j in range(0, ncols):
			axes[i][j].scatter(qc_injection_order, qc_filter_pred_met[n], color = 'red', marker ='H', alpha = alpha_q )
			axes[i][j].scatter(sample_injection_order, sample_filter_pred_meta[n], color = 'blue', alpha = alpha_s)
			axes[i][j].legend(['QC', 'Sample'], loc = 'best')
			axes[i][j].set_ylim(xlim, ylim)
			n += 1
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()


def prepare_data_to_gaussian_distribution(ratio_filter_qc, ratio_filter_sample, qc_inj_ord, sample_inj_ord):
	filter_qc_ratio_distribution = abs(np.array(ratio_filter_qc) - 1)
	filter_sample_ratio_distribution = abs(np.array(ratio_filter_sample) - 1)
	distribution = []
	for i in range(0, 50):
		qc = {}
		sample = {}
		for j in range (0, len(qc_inj_ord)):
			qc[qc_inj_ord[j]] = filter_qc_ratio_distribution[i][j]
		for j in range(0, len(sample_inj_ord)):
			sample[sample_inj_ord[j]] = filter_sample_ratio_distribution[i][j]
		distribution.append(sorted((qc | sample).items()))
	return distribution



def subplots_gaussian_distribution(distrib, title, xsize, ysize, nrows = 10, ncols = 5, sharex= False, kde = True, fig_save=False, fig_path =None):
	fig, axes = plt.subplots(nrows =nrows, ncols= ncols, sharex = sharex, figsize=(xsize, ysize))
	fig.suptitle(title)
	n = 0
	for i in range(0, nrows):
		for j in range(0, ncols):
			sb.histplot(list(dict(distrib[n]).values()), ax=axes[i][j], kde = kde)
			n += 1
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()


def subplots_fi(fi_list, title, legend, xsize, ysize, nrows, ncols):
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(xsize, ysize))
	fig.suptitle(title)
	n = 0
	for i in range(0, nrows):
		for j in range(0, ncols):
			sb.barplot(x=fi_list[n][legend[0]], y=fi_list[n][legend[1]], ax=ax[i][j])
			ax[i][j].legend(legend, loc = 'best')
			n += 1

def plot_features_importances(title, fi, xlabel, ylabel, xsize, ysize,fig_save = False, fig_path = None):
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	fi_plot = sb.barplot(x=fi[xlabel], y=fi[ylabel])
	fi_plot.set_xticklabels(fi[xlabel], rotation = 60)
	if fig_save:
		plt.savefig(fig_path, format='svg')
	plt.show()


def count_tuples_occurrences(keys, parameters):
	tuples = []
	quantity = []
	for p in parameters:
		tuple = []
		found = False
		for key in keys:
			tuple.append(p[key])
		for i in range(0, len(tuples)):
			if tuple == tuples[i]:
				quantity[i] +=1
				found = True
				break
		if not found:
			tuples.append(tuple)
			quantity.append(1)
	return tuples, quantity



def plot_grid_best_parameters(title, grid_search, xsize, ysize, labels_separator='\n', keys_labels=True, save_fig = False, fig_path = None):
	grid_parameters = grid_search[0].param_grid
	keys = []
	values = []
	for key in grid_parameters.keys():
		keys.append(key)
	for grid in grid_search:
		values.append(grid.best_params_)
	tuples, quantity = count_tuples_occurrences(keys, values)
	labels = []
	for t in tuples:
		label = ''
		for elem in range(len(t)):
			if elem != len(t)-1:
				if keys_labels:
					label += str(keys[elem] + ": " + str(t[elem]) + labels_separator)
				else:
					label += str(t[elem]) + labels_separator
			else:
				if keys_labels:
					label += str(keys[elem]) + ": " + str(t[elem])
				else:
					label += str(t[elem])
		labels.append(label)
	plt.figure(figsize=(xsize, ysize))
	plt.title(title)
	sb.barplot(x=labels, y=quantity, order=quantity.sort())
	if save_fig: 
		plt.savefig(fig_path, format= 'svg')
	plt.show()
