import matplotlib as mpl
import matplotlib.pyplot as plt
import json
# from sklearn.matrics import plot_confution_matrix
import itertools
import seaborn as sn
import pandas as pd

import numpy as np

def plot_results():
    with open('results/results.json', 'r') as f:
        bleu = json.load(f)[0: 140]

    with open('results/loss.json', 'r') as f:
        loss = json.load(f)[0: 140]


    x = [ i * 5000 for i in range(140) ]

    print(x)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Bleu Score')
    ax.set_xlabel('Iter')
    ax.set_ylabel('Bleu Score')
    ax.plot(x, bleu)
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Loss')
    ax.set_xlabel('Iter')
    ax.set_ylabel('Loss')
    ax.plot(x, loss)
    plt.show()
    

    
if __name__ == '__main__':
    plot_results()
