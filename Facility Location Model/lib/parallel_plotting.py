"""
Modified by TRomijn from Jan H. Kwakkel and/or Sibel Eker
Forked from github/quaquel/epa1361_open to github/TRomijn/epa1361_open

Last changed by TRomijn on 2 May 2018
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import numpy as np

def setup_parallel_plot(labels, maxima, minima):
    #labels is a list, minima and maxima pd series
    nr_columns = len(labels)
    fig = plt.figure()
    axes = []
    fs = 16
    # we need one axes less than the shape
    for i, label in enumerate(labels[:-1]):
        i += 1
        ax = fig.add_subplot(1,nr_columns-1,i,  ylim=(-0.1,1.1))
        axes.append(ax)
        ax.set_xlim([i,i+1])
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        ax.xaxis.set_ticklabels([labels[i-1]], rotation=45, fontsize=fs)
        ax.xaxis.set_tick_params(bottom=False, top=False)
        
        #let's put our own tick labels
        ax.yaxis.set_ticks([])
        ax.text(i, 1.01, "{:.2f}".format(maxima[label]), va="bottom", ha="center", fontsize=fs)
        ax.text(i, -0.01,"{:.2f}".format(minima[label]), va="top", ha="center", fontsize=fs)
        
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['right'].set_bounds(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    
    # for the last axis, we need 2 ticks (also for the right hand side
    ax.spines['right'].set_bounds(0, 1)
    ax.xaxis.set_major_locator(ticker.FixedLocator([i, i+1]))
    ax.xaxis.set_ticklabels(labels[i-1:i+1], fontsize=fs, rotation=45)
    ax.text(i+1, 1.01, "{:.2f}".format(maxima[labels[-1]]), va="bottom", ha="center", fontsize=fs)
    ax.text(i+1, -0.01,"{:.2f}".format(minima[labels[-1]]), va="top", ha="center", fontsize=fs)
    
    # add the tick labels to the rightmost spine
    for tick in ax.yaxis.get_major_ticks():
        tick.label2On=True
    
    # stack the subplots together
    plt.subplots_adjust(wspace=0)
    
    return axes


def normalize(data, minima, maxima):
    #takes pandas dataframe as data, and series as minima and maxima
    d = maxima - minima
    d[d==0] = 1
    norm_data = data.copy()
    for col in data.columns:
        norm_data[col] = (data[col]-minima[col])/d[col]
    #norm_data = data/d - minima/d
    return norm_data

def plot_optimal(data, labels, title):
    # print(plt.__version__)
    minima = np.min(data, axis=0)
    maxima = np.max(data, axis=0)
    #print(maxima)
    axes = setup_parallel_plot(labels, maxima, minima)

    normed_data = normalize(data, minima, maxima)
    #normed_data = normed_data.sort_index(axis=1) #re-ordering the columns of the df to align the grid with the df

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for j in range(len(labels)-1):
        ax = axes[j]
        y = normed_data.iloc[:, j:j+2]  
        x = np.tile([j+1,j+2], (y.shape[0], 1))
        ax.plot(x.T, y.T, color='grey', alpha=0.7, linewidth=0.7)
    
#     change_fontsize(fig, 14)
    fig.suptitle(title, fontsize=16)
    plt.show()  