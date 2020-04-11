import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def read_bci_data():
    S4b_train = np.load('data/S4b_train.npz')
    X11b_train = np.load('data/X11b_train.npz')
    S4b_test = np.load('data/S4b_test.npz')
    X11b_test = np.load('data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

def read_bci_train_data():
    S4b_train = np.load('data/S4b_train.npz')
    X11b_train = np.load('data/X11b_train.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)

    train_label = train_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    print('training set:', train_data.shape, train_label.shape)

    return torch.from_numpy(train_data).to(device), torch.from_numpy(train_label).to(device)

def read_bci_test_data():
    S4b_test = np.load('data/S4b_test.npz')
    X11b_test = np.load('data/X11b_test.npz')

    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    test_label = test_label -1
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print('testing set:', test_data.shape, test_label.shape)

    return torch.from_numpy(test_data).to(device), torch.from_numpy(test_label).to(device)
