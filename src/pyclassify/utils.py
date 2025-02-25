import os
import yaml
from line_profiler import profile
import numpy as np
import numba
from numba.pycc import CC
# both import statements are needed in the read_config function



@profile
def distance(point1: list[float], point2: list[float]) -> float:
    """
    This function computes the square of the Euclidean distance between two points, point1 and point2.
    Inputs:
        point1: list[float] the first point
        point2: list[float] the second point
    Returns: float (distance between the points)
    """
    assert len(point1)==len(point2), f"Points have different dimensions: {len(point1)} != {len(point2)}"
    return sum((x - y) ** 2 for x, y in zip(point1, point2))



@profile
def distance_numpy(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Same as 'distance', but using numpy arrays
    """
    assert point1.shape==point2.shape,  f"Points have different dimensions: {point1.shape} != {point2.shape}"
    difference = point1 - point2
    square_norm = np.sum(difference ** 2) # float(np.dot(difference, difference))
    return square_norm



cc = CC("distance_numba")
@cc.export("distance_numba", "f8(f8[:], f8[:])")
@numba.njit(nogil=True, parallel=True)
def distance_numba(point1, point2):
    """
    Same as the other distance functions, but implemented in numba.
    Remark that adding the profile decorator before the ones involving numba results in a warning.
    Moreover, adding it after the numba decorator causes compilation errors.
    """
    assert point1.shape == point2.shape, f"Points have different dimensions: {point1.shape} != {point2.shape}"
    difference = point1 - point2
    square_norm = float(np.dot(difference, difference))
    return square_norm



@profile
def majority_vote(neighbors: list[int]) -> int:
    """
    This class returns the most common class (expressed as an integer) in a given list of neighbor labels.
    Input:
        neighbors: list[int] (the labels of the neighbors)
    Returns:
        int (result of majority voting)
    """
    return max(set(neighbors), key=neighbors.count)



def read_config(file: str) -> dict:
    """
    To read the desired configuration file, passed in input as a string
    Input:
        file: str (representing the location of file)
    Returns:
        dict (containing the configuration parameters needed)
    """
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs



def read_file(dataset: str) -> tuple[list[list[float]], list[int]]:
    """
    Function used to read the dataset and to return X, y
    Input:
        dataset: str representing the location of the dataset
    Returns:
        tuple[list[list[float]], list[int]] (the dataset structured as tuple X,y )
    """
    lines = []
    with open (dataset, "r") as data:
        lines = [line.strip(' ').split(',') for line in data]
    
    X = []
    y = []
    
    for line in lines:
        X.append([float(x_j) for x_j in line[:-1]])
        # these cheks are made to ensure the function works with both datasets under consideration
        if line[-1][0]=='g':
            y.append(0)
        elif line[-1][0]=='b':
            y.append(1)
        else:
            y.append(int(line[-1]))
    return X, y



if __name__ == "__main__":
    cc.compile()
    print("Compilation completed.")
