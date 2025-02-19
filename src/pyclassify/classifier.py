from .utils import distance, majority_vote
# or, instead, from pyclassify import ...
# pyclassify already assumes that pyclassify is installed, the previous one does not



class kNN():
    """
    Class that implements a kNN classifier.
    """
    def __init__(self, k: int):
        if not isinstance(k, int) or k<=0:
            raise TypeError("k must be a natural number")
        self.k = k


    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x: list[float]) -> list[int]:
        """
        Method to get the k nearest neighbors
        """
        enumerated_distances = list(enumerate([distance(X_i, x) for X_i in X]))
        sorted_distances = sorted(enumerated_distances, key=lambda x: x[1])        

        k_nearest_neighbors = [index for index, _ in sorted_distances[:self.k]]
        y_labels = [y[index] for index in k_nearest_neighbors]
        return y_labels


    def __call__(self, data: tuple[list[list[float]], tuple[int]], new_points: list[list[float]]) -> list[int]:
        """
        This method returns the predicted labels of a new set of points new_points given a training dataset data=(X, y)
        """
        predicted_labels = []

        for point in new_points:
            closest_neighbors = self._get_k_nearest_neighbors(data[0], data[1], point)
            predicted_label = majority_vote(closest_neighbors)
            predicted_labels.append(predicted_label)
        return predicted_labels
