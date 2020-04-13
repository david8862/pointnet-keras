#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt to plot 3D pointcloud chart on ModelNet40 dataset
'''
import os, argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
               'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
               'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

def load_h5(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def plot_cloud_cls(datafile_path, sample_index):
    sample_points, sample_labels = load_h5(datafile_path)
    sample = sample_points[sample_index]
    label = int(sample_labels[sample_index])

    x = sample[:, 0]
    y = sample[:, 1]
    z = sample[:, 2]

    # plot the scatter chart
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(x, y, z, label=class_names[label])

    # add axis label (order: Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.legend(loc='lower right')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot 3D pointcloud chart on ModelNet40 dataset')
    parser.add_argument('--datafile_path', type=str, required=True, help='loaded ModelNet40 h5 datafile')
    parser.add_argument('--sample_index', type=int, required=False, help='data sample index to show, default=0', default=0)
    args = parser.parse_args()

    plot_cloud_cls(args.datafile_path, args.sample_index)


if __name__ == "__main__":
    main()
