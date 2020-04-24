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
        results = json.load(f)

    x = [ i+1 for i in range(15) ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle(color=[
                          '#F7C242',
                          '#D9AB42',
                          '#85A7C2',
                          '#004368'])
    ax.set_title('ResNet18')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.plot(x, results['resnet18_pretrained']['train_acc'][0:15], alpha=0.6, label='pretrained_train')
    ax.plot(x, results['resnet18_pretrained']['test_acc'][0:15], alpha=0.6, label='pretrained_test')
    ax.plot(x, results['resnet18']['train_acc'][0:15], alpha=0.6, label='train')
    ax.plot(x, results['resnet18']['test_acc' ][0:15], alpha=0.6, label='test')
    ax.legend(loc='lower right')
    plt.savefig('results/imgs/resnet18.png')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle(color=[
                          '#F7C242',
                          '#D9AB42',
                          '#85A7C2',
                          '#004368'])
    ax.set_title('ResNet50')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.plot(x, results['resnet50_pretrained']['train_acc'][0:15], alpha=0.6, label='pretrained_train')
    ax.plot(x, results['resnet50_pretrained']['test_acc'][0:15], alpha=0.6, label='pretrained_test')
    ax.plot(x, results['resnet50']['train_acc'][0:15], alpha=0.6, label='train')
    ax.plot(x, results['resnet50']['test_acc' ][0:15], alpha=0.6, label='test')
    ax.legend(loc='lower right')
    plt.savefig('results/imgs/resnet50.png')

def plot_confusion_matrix():
    resnet18_pre = [[4995, 5, 149, 0, 4], [395, 8, 84, 0, 1], [390, 6, 637, 30, 19], [13, 0, 78, 68, 16], [21, 0, 36, 6, 64]]
    resnet50_pre = [[4995, 1, 155, 0, 2], [400, 2, 85, 0, 1], [405, 3, 632, 31, 11], [11, 0, 89, 66, 9], [23, 0, 35, 12, 57]]
    
    resnet18_pre = np.array(resnet18_pre)
    cnt = np.sum(resnet18_pre)
    df_cm = pd.DataFrame(resnet18_pre / cnt, index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 16}, cmap=sn.cubehelix_palette(8))
    plt.xlabel('outputs')
    plt.ylabel('labels')
    plt.title('ResNet18 (Pretrained)')
    # plt.show()
    plt.savefig('results/imgs/resnet18_confusion.png')
    
    resnet50_pre = np.array(resnet50_pre)
    cnt = np.sum(resnet50_pre)
    df_cm = pd.DataFrame(resnet50_pre / cnt, index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 16}, cmap=sn.cubehelix_palette(8))
    plt.xlabel('outputs')
    plt.ylabel('labels')
    plt.title('ResNet50 (Pretrained)')
    # plt.show()
    plt.savefig('results/imgs/resnet50_confusion.png')
    


    
if __name__ == '__main__':
    plot_results()
    # plot_confusion_matrix()
