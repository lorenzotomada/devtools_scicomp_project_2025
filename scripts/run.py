from pyclassify import kNN
from pyclassify.utils import read_config, read_file
import argparse


print('\nThis script is supposed to be executed while in the main folder, using the bash command "python scripts/run.py --config=experiments/config".\n\nIf the config arugument is not passed, a default value is provided here, reading the parameters in the "config.yaml" file.\n')


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="config file:")


args = parser.parse_args()
filename = args.config if args.config else './experiments/config'


kwargs = read_config(filename)
dataset = kwargs["dataset"]
X, y = read_file(dataset)


k = kwargs["k"]
knn_classifier = kNN(k)


n_samples = len(X)
test_fraction = 0.8
train_fraction = 1 - test_fraction


first_test_index = int(train_fraction*n_samples)

X_train = X[0: first_test_index]
X_test = X[first_test_index:]

y_train = y[0: first_test_index]
y_test = y[first_test_index:]

y_pred = knn_classifier((X_train, y_train), X_test)


errors = sum(1 for (true_label, predicted_label) in zip(y_test, y_pred) if true_label!=predicted_label)
print(f'Accuracy of the kNN classifier with k = {k}: {100*(1-errors/len(y_test)):.4f}%\n')
