import matplotlib as mpl
import matplotlib.pyplot as plt
import json
# from sklearn.matrics import plot_confution_matrix
import itertools
import seaborn as sn
import pandas as pd

import numpy as np

def plot_results():

    x = [ i * 3 for i in range(10) ]
    monotonic_CEloss = [21.3, 20.0, 18.3, 17.7, 18.3, 16.5, 16.4, 16.5, 16.4, 16.4]
    monotonic_KLloss = [0.016, 0.0025, 0.03, 0.00012, 0.0008, 0.0007, 0.0008, 0.0003, 0.0002, 0.0002]
    monotonic_bleu   = [0.04, 0.04, 0.04, 0.02, 0.03, 0.04, 0.04, 0.05, 0.05, 0.04]
    
    cyclical_CEloss = [19.7, 18.9, 17.7, 16.5, 12.5, 11.9, 9.3, 9.7, 11.1, 10.2]
    cyclical_KLloss = [0.016, 1.2, 33.78, 24.02, 21.73, 28.22, 34.83, 33.53, 35.66, 34.44]
    cyclical_bleu   = [0.04, 0.12, 0.22, 0.25, 0.17, 0.48, 0.46, 0.57, 0.50, 0.42]


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Crossentropy Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(x, monotonic_CEloss, label='monotonic')
    ax.plot(x, cyclical_CEloss, label='cyclical')
    ax.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('KL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(x, monotonic_KLloss, label='monotonic')
    ax.plot(x, cyclical_KLloss, label='cyclical')
    ax.legend(loc='upper right')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('BLEU-4 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.plot(x, monotonic_bleu, label='monotonic')
    ax.plot(x, cyclical_bleu, label='cyclical')
    ax.legend(loc='lower right')
    plt.show()
    
if __name__ == '__main__':
    plot_results()
