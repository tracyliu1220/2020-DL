import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    def _gen_circle(n=100):
        inputs = []
        labels = []

        for i in range(n):
            for j in range(n):
                x = (i + 1) / (n + 1)
                y = (j + 1) / (n + 1)
                inputs.append([x, y])
                if abs(x - 0.5) * abs(x - 0.5) + abs(y - 0.5) * abs(y - 0.5)  <= 0.09:
                    labels.append(1)
                else:
                    labels.append(0)
        
        return np.array(inputs), np.array(labels).reshape((-1, 1))

    def _gen_grid(n=100):
        inputs = []
        labels = []
        
        for i in range(n):
            for j in range(n):
                x = (i + 1) / (n + 1)
                y = (j + 1) / (n + 1)
                inputs.append([x, y])
                labels.append((int(x / 0.2) % 2) ^ (int(y / 0.2) % 2))
        
        return np.array(inputs), np.array(labels).reshape((-1, 1))


    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR' or mode == 'Circle' or mode == 'Grid'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor,
            'Circle': GenData._gen_circle,
            'Grid': GenData._gen_grid
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100, lr=0.2): #TODO
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval

        # Model parameters initialization
        # Please initiate your network parameters here.
        input_size = 2
        output_size = 1
        self.lr = lr
        self.mo = 0.9
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size)
        self.w3 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))
        
        self.v_w1 = np.zeros((input_size, hidden_size) )
        self.v_w2 = np.zeros((hidden_size, hidden_size))
        self.v_w3 = np.zeros((hidden_size, output_size))
        self.v_b1 = np.zeros((1, hidden_size))
        self.v_b2 = np.zeros((1, hidden_size))
        self.v_b3 = np.zeros((1, output_size))

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs): #TODO
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """
        self.input = inputs
        self.a1    = sigmoid(np.dot(self.input, self.w1) + self.b1)
        self.a2    = sigmoid(np.dot(self.a1, self.w2) + self.b2)
        output     = sigmoid(np.dot(self.a2, self.w3) + self.b3)

        return output

    def backward(self): #TODO
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        dout    = self.error
        dout    = np.multiply(dout, der_sigmoid(self.output))
        grad_w3 = np.dot(self.a2.T, dout)
        grad_b3 = np.sum(dout, axis=0)

        dout    = np.dot(dout, self.w3.T)
        dout    = np.multiply(dout, der_sigmoid(self.a2))
        grad_w2 = np.dot(self.a1.T, dout)
        grad_b2 = np.sum(dout, axis=0)

        dout    = np.dot(dout, self.w2.T)
        dout    = np.multiply(dout, der_sigmoid(self.a1))
        grad_w1 = np.dot(self.input.T, dout)
        grad_b1 = np.sum(dout, axis=0)
        
        
        self.v_w1 = self.mo * self.v_w1 + self.lr * grad_w1 
        self.v_w2 = self.mo * self.v_w2 + self.lr * grad_w2
        self.v_w3 = self.mo * self.v_w3 + self.lr * grad_w3
                                                           
        self.v_b1 = self.mo * self.v_b1 + self.lr * grad_b1
        self.v_b2 = self.mo * self.v_b2 + self.lr * grad_b2
        self.v_b3 = self.mo * self.v_b3 + self.lr * grad_b3

        self.w1 -= self.v_w1
        self.w2 -= self.v_w2
        self.w3 -= self.v_w3
                            
        self.b1 -= self.v_b1
        self.b2 -= self.v_b2
        self.b3 -= self.v_b3

        return

    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        self.pre_error = 100000
        error = 0

        for epochs in range(self.num_step):
            error = 0

            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx:idx+1, :]
                self.backward()

                error += self.error[0][0] * self.error[0][0]

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)
        
            
            if error > self.pre_error:
                self.lr *= 0.8
                pass
            self.pre_error = error

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]
        error = 0
        acc = 0

        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])
            acc += (result[0][0] >= 0.5) == labels[idx:idx+1, :][0][0]

        error /= n
        acc /= n
        print('accuracy: %.2f' % (acc * 100) + '%')
        print('')


if __name__ == '__main__':
    data, label = GenData.fetch_data('XOR', 100)

    net = SimpleNet(, num_step=1000, lr=0.1)
    net.train(data, label)
    
    pred_result = np.round(net.forward(data))
    SimpleNet.plot_result(data, label, pred_result)

    data, label = GenData.fetch_data('XOR', 150)
    net.test(data, label)

    pred_result = np.round(net.forward(data))
    SimpleNet.plot_result(data, label, pred_result)
