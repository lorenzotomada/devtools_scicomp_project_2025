from .utils import distance, distance_numpy, majority_vote
from .distance_numba import distance_numba
from line_profiler import profile
import numpy as np
# or, instead, from pyclassify import ...
# pyclassify already assumes that pyclassify is installed, the previous one does not



class kNN():
    """
    Class that implements a kNN classifier.
    Inputs of the constructor:
        int: k (number of neighbors to consider)
        str: backend (optional, needed to chose the distance function. It can be either 'plain', 'numpy' or 'numba')
    """
    def __init__(self, k: int, backend="plain"):
        if not isinstance(k, int) or k<=0:
            raise TypeError("k must be a natural number")
        if backend not in ("plain", "numpy", "numba"):
            raise ValueError("Invalid value of backend: must be either 'numpy' or 'plain'")
        self.k = k
        self.distance = distance_numpy if backend=='numpy' else distance_numba if backend=='numba' else distance


    @profile
    def _get_k_nearest_neighbors(self, X, y, x):
        """
        Method to get the k nearest neighbors of the point x.
        Inputs:
            X: list[lists[float]] or list[np.array] the list of points composing the dataset
            x: list[float] or np.array (consistently with X). This is the point of which we want the neighbors
            y: list[int] a vector containing the labels corresponding to points of X
        """
        enumerated_distances = list(enumerate([self.distance(X_i, x) for X_i in X]))
        sorted_distances = sorted(enumerated_distances, key=lambda x: x[1])        

        k_nearest_neighbors = [index for index, _ in sorted_distances[:self.k]]
        y_labels = [y[index] for index in k_nearest_neighbors]
        return y_labels


    @profile
    def __call__(self, data: tuple[list[list[float]], list[int]], new_points: list[list[float]]) -> list[int]:
        """
        This method returns the predicted labels of a new set of points new_points given a training dataset data=(X, y).
        Inputs:
            data:  tuple[list[list[float]], list[int]] formed by the couple (X, y) of which the dataset consists
            new_points: list[list[float]] points which we want to classify
        Returns: list[int] the predicted labels corresponding to new_points
        """
        predicted_labels = []

        if self.distance in (distance_numpy, distance_numba) :
            X_data = [np.array(X_i) for X_i in data[0]]
            _new_points= [np.array(point) for point in new_points] # created for conversion to np arrays
        else:
            X_data = data[0]
            _new_points = new_points
        y_data = data[1]

        for point in _new_points:
            closest_neighbors = self._get_k_nearest_neighbors(X_data, y_data, point)
            predicted_label = majority_vote(closest_neighbors)
            predicted_labels.append(predicted_label)

        return predicted_labels
