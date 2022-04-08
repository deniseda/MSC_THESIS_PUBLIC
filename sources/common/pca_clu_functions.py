from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from matrix import *
import itertools



def do_PCA(matrix, components = 2):
	pca = PCA(n_components= components)
	pca.fit_transform(matrix)
	component = pca.components_
	variance = pca.explained_variance_ratio_
	return component, variance


def plot_explained_variance(component, variance, title, xstart, xstop, xstep, ystart, ystop, ystep, xsize, ysize, linewidth=2, color = 'red', marker = 'o', save_fig=False, fig_path=None):
	plt.figure(figsize=(xsize, ysize))
	pca_data = np.arange(component.shape[0]) + 1
	plt.plot(pca_data, variance, linewidth=linewidth, color=color, marker=marker)
	plt.title(title)
	plt.xlabel('Principal components')
	plt.ylabel('Variance explained')
	plt.xticks(np.arange(xstart, xstop, xstep))
	plt.yticks(np.arange(ystart, ystop, ystep))
	if save_fig:
		plt.savefig(fig_path, format='svg')
	plt.show()


def dataframe_builder(list_labels, columns):
	df = pd.DataFrame()
	for i in range(0, len(list_labels)):
		df[list_labels[i]] = columns[i]
	return df

	
def plot_2dimensions(df, title, xsize, ysize, save_fig=False, fig_path=None):
	plt.figure(figsize=(xsize, ysize))
	sb.scatterplot(data = df, x= 'PC1', y = 'PC2', hue= 'Labels')
	plt.title(title)
	if save_fig:
		plt.savefig(fig_path, format='svg')
	plt.show()


def plot_3dimensions(df, title, save_fig=False, fig_path=None):
	fig = px.scatter_3d(df, x ='PC1', y= 'PC2', z = 'PC3', color = 'Labels')
	fig.update_layout(title_text=(title), title_x=0.5)
	if save_fig:
		fig.write_image(fig_path)
	fig.show()



def order_labels(dataframe):
	sort_pca_df = cluster(dataframe)
	all_princ_comp_sorted = sort_pca_df.drop(['Labels'], axis = 1, inplace= False)
	sort_labels_df = sort_pca_df.sort_values(by= 'Labels').Labels
	return all_princ_comp_sorted, sort_labels_df


def heatmap_plot(dataf_pairdist, xlabels, ylabels, xsize, ysize, title, cmap = 'coolwarm', center=None , annot = False, save_fig=False, fig_path=None, vmin = None, vmax = None):
	plt.figure(figsize=(xsize, ysize))
	sb.heatmap(dataf_pairdist, xticklabels= xlabels, yticklabels = ylabels,annot= annot, cmap = cmap, center = center, vmin=vmin, vmax = vmax)
	plt.title(title)
	if save_fig:
		plt.savefig(fig_path, format='svg')
	plt.show()



def plot_intergroup_distance(mean, format, title, xlabel, ylabel, xsize, ysize, cmap='coolwarm', center=None, save_fig= False, fig_path= None, vmin= None, vmax= None):
	fig, ax = plt.subplots(figsize=(xsize, ysize))
	ax = sb.heatmap(mean, annot= format, fmt='', xticklabels= xlabel, yticklabels= ylabel, cmap = cmap, vmin=vmin, vmax= vmax)
	fig.suptitle(title)
	if save_fig:
		plt.savefig(fig_path, format='svg')
	plt.show()




def do_tsne(matrix, components= 2, verbose= 1, threads = -1):
	tsne = TSNE(n_components= components, verbose = verbose , n_jobs= threads)
	tsne.fit_transform(matrix)
	components_tsne = tsne.embedding_
	return components_tsne

### intergroup distance

def get_group_indices(labels_list, label):
	for i in range(len(labels_list)):
		if labels_list[i] == label:
			begin = i
			break
	end = len(labels_list)
	for i in range(begin, len(labels_list)):
		if labels_list[i] != label:
			end = i
			end -= 1
			break
	return begin, end


def intergroup_distance(distances, labels, group1, group2):
	init_ind_gr1, end_ind_gr1 = get_group_indices(labels_list=labels, label=group1)
	init_ind_gr2, end_ind_gr2 = get_group_indices(labels_list=labels, label= group2)
	submatrix = distances[init_ind_gr1 : end_ind_gr1, init_ind_gr2 : end_ind_gr2]
	return np.mean(submatrix), np.std(submatrix) 


def all_intergroup_distances(distances, labels, group_list):
	permutation = list(itertools.product(group_list, repeat = 2))
	mean_matrix = np.zeros((len(group_list), len(group_list)))
	variance_matrix = np.zeros((len(group_list), len(group_list)))
	n = 0
	for i in range(mean_matrix.shape[0]):
		for j in range(mean_matrix.shape[1]):
			mean , variance = intergroup_distance(distances, labels, permutation[n][0], permutation[n][1])
			mean_matrix[i][j] = mean
			variance_matrix[i][j] = variance
			n += 1
	return mean_matrix, variance_matrix