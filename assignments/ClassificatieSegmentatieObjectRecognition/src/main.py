from cgi import test
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_digits


# Copied from stackoverflow because I can't get numpy to zip arrays in a normal way.
# https://stackoverflow.com/a/4602224
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == "__main__":
    digits = load_digits()
    data, target = unison_shuffled_copies(digits.data, digits.target)

    # Create training and testing dataseta
    test_size = len(data) // 3
    train_size = len(data) - test_size
    X, y = data[:-test_size], target[:-test_size]
    test_X, test_y = data[-test_size:], target[-test_size:]

    print(f"Train dataset size: {train_size}")
    print(f"Test dataset size: {test_size}")

    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(X, y)

    test_correct = 0
    res = clf.predict(test_X)

    for i in range(test_size):
        if res[i] == test_y[i]:
            test_correct += 1

    print(f"Score: {test_correct}/{test_size} ({test_correct/test_size*100:.2f}%)")
