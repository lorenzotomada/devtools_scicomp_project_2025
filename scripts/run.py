from pyclassify import kNN
from pyclassify.utils import read_config, read_file
import numpy as np
import random
import argparse


random.seed(226)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")


args = parser.parse_args()
filename = args.config if args.config else './experiments/config' # automatic choice if no argument is passed


kwargs = read_config(filename)
dataset = kwargs["dataset"]
X, y = read_file(dataset)


k = kwargs["k"]
backend = kwargs.get("backend", "plain")
print(f"The selected backend is {backend}.")


knn_classifier = kNN(k, backend)


n_samples = len(X)
test_fraction = 0.8
train_fraction = 1 - test_fraction


indices = [i for i in range(n_samples)]
random.shuffle(indices) 


first_test_index = int(train_fraction*n_samples) # split in train and test
train_indices = indices[:first_test_index]
test_indices = indices[first_test_index:]


X_train = [X[i] for i in train_indices]
X_test = [X[i] for i in test_indices]


y_train = [y[i] for i in train_indices]
y_test = [y[i] for i in test_indices]


y_pred = knn_classifier((X_train, y_train), X_test) # perform the prediction


errors = sum(1 for (true_label, predicted_label) in zip(y_test, y_pred) if true_label!=predicted_label)
print(f'Accuracy of the kNN classifier with k = {k}: {100*(1-errors/len(y_test)):.2f}%\n')


print('\n########### IMPORTANT: READ BELOW ##########\nTo study scalability, please use the script scalability.sh in the "shell" folder')