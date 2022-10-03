# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

########
# Plot #
########


def create_plots(list_filenames, list_legend_names=[], loc='lower right'):
    size = 42
    size_legend = 25
    fig = plt.figure(figsize=(12, 12))

    for i in range(len(list_filenames)):
        arr = np.loadtxt(list_filenames[i], delimiter=', ')         # load data

        print(arr.shape)

        means = np.mean(arr, axis=0)                                # y-axis values
        x_axis = np.arange(means.shape[0])                          # x-axis values
        se = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])    # standard error (ddof=1 for sample)

        plt.plot(x_axis, means, label=str(list_legend_names[i]), linewidth=3.0)
        plt.fill_between(x_axis, means - se, means + se, alpha=0.2)

    # x-axis
    plt.xlim(0,arr.shape[1])

    plt.xlabel('Time Step', fontsize=size, weight='bold')
    plt.xticks(fontsize=size)
    plt.xticks(np.arange(0.0, arr.shape[1] + 100, 5000), fontsize=size)

    # y-axis
    plt.ylabel('G-mean', fontsize=size, weight='bold')
    plt.yticks(np.arange(0.0, 1.000001, 0.2), fontsize=size)
    plt.ylim(0.0, 1.0)

    # legend
    if 1:
        leg = plt.legend(ncol=1, loc=loc, fontsize=size_legend)
        leg.get_frame().set_alpha(0.9)

    # grid
    plt.grid(linestyle='dotted')

    # plot
    plt.show()

    # save
    # fig.savefig(out_dir + 'test.pdf', bbox_inches='tight')


########
# test #
########

out_dir = 'exps/'
data = 'sea10'
filenames = [
    out_dir + data + '_actisiamese_10_0.01' + '_preq_' + 'gmean' + '.txt',
]

legend = ['ActiSiamese']
create_plots(filenames, list_legend_names = legend, loc='lower right')
