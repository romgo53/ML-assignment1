import numpy as np
from scipy.spatial import distance


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {
        "k": k,
        "x_train": x_train,
        "y_train": y_train
    }

    return classifier


def predictknn(classifier, x_test: np.array):
    """
    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    
    k = classifier["k"]
    x_train = classifier["x_train"]
    y_train = classifier["y_train"].reshape(-1)
    n = x_test.shape[0]
    pred = np.zeros((n, 1), dtype=y_train.dtype)

    for i in range(n):
        distances = np.array([np.linalg.norm(x_test[i] - x_train[j]) 
                              for j in range(len(x_train))])
        k_nearest_labels = y_train[np.argsort(distances)[:k]]
        
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        pred[i] = unique_labels[np.argmax(counts)]

    return pred.reshape(-1, 1)


def predictknn2(classifier, x_test):
    k = classifier["k"]
    x_train = classifier["x_train"]
    y_train = classifier["y_train"].reshape(-1)

    test_sq  = np.sum(x_test**2,  axis=1, keepdims=True)  # (n, 1)
    train_sq = np.sum(x_train**2, axis=1)                  # (m,)
    cross    = x_test @ x_train.T                          # (n, m)
    dists    = np.sqrt(test_sq + train_sq - 2*cross)       # (n, m)

    k_nearest_indices = np.argsort(dists, axis=1)[:, :k]   # (n, k)
    k_nearest_labels  = y_train[k_nearest_indices]          # (n, k)

    unique_labels = np.unique(y_train)
    counts = np.array([(k_nearest_labels == label).sum(axis=1)
                       for label in unique_labels])         # (num_labels, n)

    return unique_labels[np.argmax(counts, axis=0)].reshape(-1, 1)  # (n, 1)

def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, tr ain1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")



if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
   




