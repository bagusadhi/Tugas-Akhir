import logging
import numpy as np
from utils.adapt_data import adapt_labels

RANDOM_SEED = 42
RNG = np.random.RandomState(42)

logger = logging.getLogger(__name__)

def get_train(label, centered=False):
    """Get training dataset for MNIST"""
    return _get_adapted_dataset("train", label, centered=centered)

def get_test(label, centered=False):
    """Get testing dataset for MNIST"""
    return _get_adapted_dataset("test", label, centered=centered)

def get_shape_input():
    """Get shape of the dataset for MNIST"""
    return (None, 64, 128, 1)

def get_shape_input_flatten():
    """Get shape of the flatten dataset for MNIST"""
    return (None, 8192)

def get_shape_label():
    """Get shape of the labels in MNIST dataset"""
    return (None,)

def num_classes():
    """Get number of classes in MNIST dataset"""
    return 3

def _get_adapted_dataset(split, label=None, centered=False, flatten=False):
    """Gets the adapted dataset for the experiments

    Args :
            split (str): train, valid or test
            label (int): int in range 0 to 10, is the class/digit
                         which is considered outlier
            centered (bool): (Default=False) data centered to [-1, 1]
            flatten (bool): (Default=False) flatten the data
    Returns :
            (tuple): <training, testing> images and labels
    """
    data = np.load('data/datasets_bigan/target/bearing_section_00_target_2022.npz')
    x_train = np.load("data/datasets_bigan/train/x_train_bearing_section_0.npy")
    y_train = np.load("data/datasets_bigan/train/y_train_bearing_section_0.npy")
    dataset = {}

    # full_x_data = data['x_train']
    # full_y_data = data['y_train']
    # full_y_data = np.array(list(map(int, full_y_data))) #PROBLEM SOLVING

    # normal_x_data = full_x_data[full_y_data!= label]
    # normal_y_data = full_y_data[full_y_data!= label]
   
    # inds = RNG.permutation(normal_x_data.shape[0])
    # normal_x_data = normal_x_data[inds]
    # normal_y_data = normal_y_data[inds]
    
    # index = int(normal_x_data.shape[0])
    
    # training_x_data = normal_x_data[:index]
    # training_y_data = normal_y_data[:index]
 
    # testing_x_data = data['x_test']
    # testing_y_data = data['y_test']
    # testing_y_data = np.array(list(map(int, testing_y_data))) #PROBLEM SOLVING

    # inds = RNG.permutation(testing_x_data.shape[0])
    # testing_x_data = testing_x_data[inds]
    # testing_y_data = testing_y_data[inds]

#=================================================================================================================

    # full_x_data = np.concatenate([data['x_train'], data['x_test']], axis=0)
    # full_y_data = np.concatenate([data['y_train'], data['y_test']], axis=0)
    # full_y_data = np.array(list(map(int, full_y_data))) #PROBLEM SOLVING

    # normal_x_data = full_x_data[full_y_data!= label]
    # normal_y_data = full_y_data[full_y_data!= label]

    # inds = RNG.permutation(normal_x_data.shape[0])
    # normal_x_data = normal_x_data[inds]
    # normal_y_data = normal_y_data[inds]

    # index = int(normal_x_data.shape[0]*0.8)

    # training_x_data = normal_x_data[:index]
    # training_y_data = normal_y_data[:index]

    # testing_x_data = np.concatenate([normal_x_data[index:], full_x_data[full_y_data == label]], axis=0)
    # testing_y_data = np.concatenate([normal_y_data[index:], full_y_data[full_y_data == label]], axis=0)

    # inds = RNG.permutation(testing_x_data.shape[0])
    # testing_x_data = testing_x_data[inds]
    # testing_y_data = testing_y_data[inds]

# ===================================================================================================


    training_x_data = x_train
    training_y_data = y_train
    # training_x_data = training_x_data[training_y_data == label]
    # training_y_data = training_y_data[training_y_data == label]

    testing_x_data = data['x_test']
    testing_y_data = data['y_test']
    # testing_y_data = np.array(list(map(int, testing_y_data)))






    
    # testing_x_data = testing_x_data[testing_y_data == label]
    # testing_y_data = testing_y_data[testing_y_data == label]

    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split == 'train':

        dataset[key_img] = training_x_data.astype(np.float32)
        dataset[key_lbl] = training_y_data.astype(np.float32)
        # dataset[key_lbl] = adapt_labels(dataset[key_lbl], label)

    elif split == 'test':

        dataset[key_img] = testing_x_data.astype(np.float32)
        dataset[key_lbl] = testing_y_data.astype(np.float32)

    if centered:
        dataset[key_img] = dataset[key_img].astype(np.float32)
        dataset[key_img] = dataset[key_img] * 2. - 1.
        # dataset[key_lbl] = adapt_labels(dataset[key_lbl], label)

    if not flatten:
        dataset[key_img] = dataset[key_img].reshape(-1, 64, 128, 1)
        # dataset[key_lbl] = adapt_labels(dataset[key_lbl], label)

            
    # dataset[key_lbl] = adapt_labels(dataset[key_lbl], label)

    return (dataset[key_img], dataset[key_lbl])


