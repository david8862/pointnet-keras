#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt to plot 3D color-segmented pointcloud chart on shapenetcore dataset
'''
import os, argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def load_h5(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def plot_cloud_seg(datafile_path, sample_index, num_points, num_categories):
    # load data and reshape according to points number
    sample_points, sample_labels = load_h5(datafile_path)
    sample_points = sample_points.reshape(-1, num_points, 3)
    sample_labels = sample_labels.reshape(-1, num_points, num_categories)

    sample = sample_points[sample_index]
    label = sample_labels[sample_index]

    # generate color map for segment categories
    color_map = cm.rainbow(np.linspace(0, 1, num_categories))
    # assign color for each point based on category label
    color = [color_map[label[i].tolist().index(max(label[i]))] for i in range(sample.shape[0])]

    x = sample[:, 0]
    y = sample[:, 1]
    z = sample[:, 2]

    # plot the scatter chart
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(x, y, z, c=color)

    # add axis label (order: Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot 3D color-segmented pointcloud chart on shapenetcore dataset')
    parser.add_argument('--datafile_path', type=str, required=True, help='loaded shapenetcore h5 datafile')
    parser.add_argument('--sample_index', type=int, required=False, help='data sample index to show, default=0', default=0)
    parser.add_argument('--num_points', type=int, required=False, help='number of points per sample, default=1024 for Airplane', default=1024)
    parser.add_argument('--num_categories', type=int, required=False, help='number of segment categories per sample, default=4 for Airplane', default=4)
    args = parser.parse_args()

    plot_cloud_seg(args.datafile_path, args.sample_index, args.num_points, args.num_categories)


if __name__ == "__main__":
    main()
