import matplotlib as mpl
import matplotlib.pyplot as plt
import json

def plot_results():
    with open('results/results.json', 'r') as f:
        results = json.load(f)

    x = [ i+1 for i in range(1000) ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle(lw=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                      color=[
                          '#F4A7B9',
                          '#F7C242',
                          '#85A7C2',
                          '#D05A6E',
                          '#D9AB42',
                          '#004368'])
    ax.set_title('EEGNet')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.plot(x, results['EEGNet_ReLU'     ]['train_acc'][0:1000], alpha=0.6, label='relu_train')
    ax.plot(x, results['EEGNet_LeakyReLU']['train_acc'][0:1000], alpha=0.6, label='leakyrelu_train')
    ax.plot(x, results['EEGNet_ELU'      ]['train_acc'][0:1000], alpha=0.6, label='elu_train')
    ax.plot(x, results['EEGNet_ReLU'     ]['test_acc' ][0:1000], alpha=0.6, label='relu_test')
    ax.plot(x, results['EEGNet_LeakyReLU']['test_acc' ][0:1000], alpha=0.6, label='leakyrelu_test')
    ax.plot(x, results['EEGNet_ELU'      ]['test_acc' ][0:1000], alpha=0.6, label='elu_test')
    ax.legend(loc='lower right')
    plt.savefig('results/imgs/EEGNet.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle(lw=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                      color=[
                          '#F4A7B9',
                          '#F7C242',
                          '#85A7C2',
                          '#D05A6E',
                          '#D9AB42',
                          '#004368'])
    ax.set_title('DeepConvNet')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.plot(x, results['DeepConvNet_ReLU'     ]['train_acc'][0:1000], alpha=0.5, label='relu_train')
    ax.plot(x, results['DeepConvNet_LeakyReLU']['train_acc'][0:1000], alpha=0.5, label='leakyrelu_train')
    ax.plot(x, results['DeepConvNet_ELU'      ]['train_acc'][0:1000], alpha=0.5, label='elu_train')
    ax.plot(x, results['DeepConvNet_ReLU'     ]['test_acc' ][0:1000], alpha=0.5, label='relu_test')
    ax.plot(x, results['DeepConvNet_LeakyReLU']['test_acc' ][0:1000], alpha=0.5, label='leakyrelu_test')
    ax.plot(x, results['DeepConvNet_ELU'      ]['test_acc' ][0:1000], alpha=0.5, label='elu_test')
    ax.legend(loc='lower right')
    plt.savefig('results/imgs/DeepConvNet.png')



if __name__ == '__main__':
    plot_results()
