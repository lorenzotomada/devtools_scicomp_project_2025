from pyclassify import kNN
from pyclassify.utils import read_config, read_file
import numpy as np
import argparse


# This script is just used to test scalability


np.random.seed(226)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config file:")


args = parser.parse_args()
filename = args.config


kwargs = read_config(filename)
N = kwargs.get("N", 1e3) # number of samples
d = kwargs.get("d", 1e4) # dimension
k = kwargs.get("k", 1)


backend = kwargs.get("backend", "plain")
print(f"The selected backend is {backend}.")


X = np.random.rand(d, N).tolist()
y = np.random.randint(2, size=d).tolist()


knn_classifier = kNN(k, backend)


n_samples = len(X)
test_fraction = 0.8
train_fraction = 1 - test_fraction


indices = [i for i in range(n_samples)]


first_test_index = int(train_fraction*n_samples) # split in train and test
train_indices = indices[:first_test_index]
test_indices = indices[first_test_index:]


X_train = [X[i] for i in train_indices]
X_test = [X[i] for i in test_indices]


y_train = [y[i] for i in train_indices]
y_test = [y[i] for i in test_indices]


y_pred = knn_classifier((X_train, y_train), X_test) # perform the prediction