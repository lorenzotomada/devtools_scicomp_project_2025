import os
import yaml
# both import statements are needed in the read_config function



def distance(point1: list[float], point2: list[float]) -> float:
    """
    This function computes the square of the Euclidean distance between two points.
    """
    assert len(point1)==len(point2), f"Points have different dimensions: {len(point1)} != {len(point2)}"
    return sum((x - y) ** 2 for x, y in zip(point1, point2))



def majority_vote(neighbors: list[int]) -> int:
    """
    This class returns the most common class (expressed as an integer) in a given list of neighbor labels.
    """
    return max(set(neighbors), key=neighbors.count)



def read_config(file: str) -> dict:
    """
    To read the desired configuration file
    """
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs



def read_file(dataset: str) -> tuple[list[list[float]], list[int]]:
    """
    Function used to read the dataset and to return X, y
    """
    lines = []
    with open (dataset, "r") as data:
        lines = [line.strip(' ').split(',') for line in data]
    
    X = []
    y = []
    
    for line in lines:
        X.append([float(x_j) for x_j in line[:-1]])
        y.append(0 if line[-1][0]=='g' else 1) # here I am checking the first element of the string and not the
                                               # whole string itself, as the latter also contains "\n"
    return X, y
