import matplotlib as plt
import json

def plot_results():
    net_name = ['EEGNet_ReLU',
                'EEGNet_LeakyReLU',
                'EEGNet_ELU',
                'DeepConvNet_ReLU',
                'DeepConvNet_LeakyReLU',
                'DeepConvNet_ELU']
    with open('results/results.json', 'r') as f:
        results = json.load(f)

    x = [ i+1 for i in range(2000) ]

    fig, ax = plt.subplots()
    ax.plot(x, results[net_name[0]])
    plot.show()



if __name__ == '__main__':
    plot_results()
