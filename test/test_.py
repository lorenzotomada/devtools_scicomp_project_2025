from pyclassify import kNN
from pyclassify.utils import majority_vote, distance
import pytest

# all tests should have test_ b/c pytest will run all the files in the folder whose name starts with test

def test_distance():
    """
    To check whether the distance function works correctly
    """
    points = []
    points.append(([1, 0, 0, 0],  [4, -3, 0, 0]))
    points.append(([0, -2, 8],  [1, 4, 2]))
    points.append(([3], [-1]))
    distances = [18, 73, 16]
    for (_distance, point_pair) in zip(distances, points):
        assert distance(*point_pair) == pytest.approx(_distance), f"Failed for points {point_pair}"

    point_2d = [1, 3]
    point_3d = [3, 5, 6]

    with pytest.raises(AssertionError): # ensure that the funtion does not worked with inputs with mismatched dimension
        x = distance(point_2d, point_3d)



def test_majority_vote():
    """
    Ensure the majority vote works as expected
    """
    points_list = [[0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
    correct_labels = [0, 1, 0]
    for (correct_label, point) in  zip(correct_labels, points_list):
        assert(majority_vote(point)==correct_label)



def test_constructor():
    """
    Checking that the constructor works properly
    """
    knn = kNN(1)
    knn = kNN(3)
    knn = kNN(1000)
    knn = kNN(2, "plain")
    knn = kNN(4, "numpy")
    knn = kNN(3, "numba")
    with pytest.raises(TypeError):
        knn = kNN("banana")
    with pytest.raises(ValueError):
        knn = kNN(2, "banana")