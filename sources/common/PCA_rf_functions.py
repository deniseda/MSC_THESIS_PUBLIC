from cProfile import label
from turtle import color
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sb




def do_pca_rf(data, components= 2):
    pca = PCA(components)
    transf_value = pca.fit_transform(data)
    return transf_value


def plot_pca_raw_normal(raw_data, pca_1_label, pca_2_label, norm_data, title, xsize, ysize, marker1 = 'D',marker2 = 'H', color1 = 'blue', color2 = 'yellow', edgecolors= 'black', alpha1 = 0.5, alpha2 = 0.4, label1 =None, label2 = None, save_fig=False, fig_path=None):
    plt.figure(figsize=(xsize, ysize))
    plt.scatter(raw_data[pca_1_label], raw_data[pca_2_label], marker = marker1, color= color1, edgecolors = edgecolors, alpha = alpha1, label= label1)
    plt.scatter(norm_data[pca_1_label], norm_data[pca_2_label], marker= marker2, color = color2, edgecolors= edgecolors, alpha= alpha2, label = label2)
    plt.title(title)
    plt.legend()
    if save_fig:
        plt.savefig(fig_path, format='svg')
    plt.show()


# if raw_data_1 is for example, rawdata qc then raw_data_2 will be rawdata samples. The same logic for norm_data_1
# This subplots will show rawdata in the first plot and data normalized in second.


def subplots_raw_normalized(raw_data_1, raw_data_2, pca_1_label, pca_2_label, norm_data_1, norm_data_2, title1, title2, xsize_fig, ysize_fig,
                            label_sc1, label_sc2, label_sc3, label_sc4, alpha1 =1, alpha2 = .1,  marker1 = 'o',
                             color1 = 'blue', color2 = 'yellow', edgecolors = 'black', save_fig = False, fig_path = None):
    fig, axes = plt.subplots(1,2 , figsize=(xsize_fig, ysize_fig))
    axes[0].scatter(raw_data_1[pca_1_label], raw_data_1[pca_2_label], marker = marker1, color = color1, edgecolors= edgecolors, alpha= alpha1, label = label_sc1)
    axes[1].scatter(norm_data_1[pca_1_label], norm_data_1[pca_2_label], marker = marker1, color = color1, edgecolors= edgecolors, alpha = alpha1, label = label_sc2)
    axes[0].scatter(raw_data_2[pca_1_label], raw_data_2[pca_2_label], marker = marker1 , color = color2, edgecolors= edgecolors, alpha = alpha2, label = label_sc3)
    axes[1].scatter(norm_data_2[pca_1_label], norm_data_2[pca_2_label], marker= marker1, color = color2, edgecolors= edgecolors, alpha = alpha2, label= label_sc4)
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    #axes[0].set_ylim(min, max)
    #axes[1].set_ylim(min, max)
    axes[0].legend()
    axes[1].legend()
    if save_fig:
        plt.savefig(fig_path, format='svg')
    plt.show()