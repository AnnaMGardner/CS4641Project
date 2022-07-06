import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

exampleData2D = pd.read_csv("/Users/lukez/Desktop/CS4641/Untitled Folder/processed_data/UnbTsne2D.csv")
exampleData3D = pd.read_csv("/Users/lukez/Desktop/CS4641/Untitled Folder/UnbTsne3D.csv")
exampleIndex = np.loadtxt("/Users/lukez/Desktop/CS4641/Untitled Folder/processed_data/unBalancedKmean.txt", dtype=int)


def largeDimensionClusterPlot2D(data2D, index):
    """
    :param data2D: pandas dataframe processed 2D data in the follow format:
            feature 1   feature 2   stroke
        0   value       value       0 or 1
        1   value       value       0 or 1
        ...
        n   value       value       0 or 1
    where n is the number of data points

    :param data2D: (nx1) numpy array of the clustering index
    ...
    """
    colors = ['red', 'orange', 'green', 'yellow', 'blue', 'pink', 'violet']

    data2D['idx'] = index
    feature1 = list(data2D.columns)[0]
    feature2 = list(data2D.columns)[1]

    clusterGroup = data2D.groupby('idx')
    clusters = [clusterGroup.get_group(x) for x in clusterGroup.groups]
    clusterIdx = 0
    for cluster in clusters:
        positive = cluster.loc[cluster['labels'] == 1]
        negative = cluster.loc[cluster['labels'] == 0]
        plt.scatter(positive[feature1], positive[feature2], marker='.', c=colors[clusterIdx])
        plt.scatter(negative[feature1], negative[feature2], marker='x', c=colors[clusterIdx])
        clusterIdx += 1
    # plt.legend()
    plt.show()
    return None


def largeDimensionClusterPlot3D(data3D, index, rotate = False):
    """
    :param data2D: pandas dataframe processed 2D data in the follow format:
            feature 1   feature 2   feature 3   stroke
        0   value       value       value       0 or 1
        1   value       value       value       0 or 1
        ...
        n   value       value       value       0 or 1
    where n is the number of data points

    :param data2D: (nx1) numpy array of the clustering index
    ...
    """
    colors = ['red', 'orange', 'green', 'yellow', 'blue', 'pink', 'violet']

    data3D['idx'] = index
    feature1 = list(data3D.columns)[0]
    feature2 = list(data3D.columns)[1]
    feature3 = list(data3D.columns)[2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    clusterGroup = data3D.groupby('idx')
    clusters = [clusterGroup.get_group(x) for x in clusterGroup.groups]
    clusterIdx = 0
    for cluster in clusters:
        # print("In cluster ", clusterIdx)
        # print(cluster['labels'].value_counts())

        positive = cluster.loc[cluster['labels'] == 1]
        negative = cluster.loc[cluster['labels'] == 0]
        ax.scatter3D(positive[feature1], positive[feature2], positive[feature3], marker='.', c=colors[clusterIdx])
        ax.scatter3D(negative[feature1], negative[feature2], negative[feature3], marker='x', c=colors[clusterIdx])
        clusterIdx += 1
    # plt.legend()
    if (rotate == True):
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.pause(.001)
    plt.show()
    return None


# largeDimensionClusterPlot2D(exampleData2D, exampleIndex)
largeDimensionClusterPlot3D(exampleData3D, exampleIndex, rotate=True)

