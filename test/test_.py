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
        # test correct computation
        assert distance(*point_pair) == pytest.approx(_distance), f"Failed for points {point_pair}"
        # test symmetry
        assert distance(point_pair[0], point_pair[1])==distance(point_pair[1], point_pair[0]), "The distance is not symmetric"
        # test distance is positive
        assert distance(point_pair[0], point_pair[0])>=0, "The distance is smaller than 0"
    # test triangle inequality. Actually here I import math and test it on sqrt(distance), since the
    # square of the L^2 norm is not a distance and does not satisfy the triangle inequality
    point1 = [1, 3, 5]
    point2 = [2, -4, 1]
    point3 = [6, 3,-2]
    from math import sqrt
    assert sqrt(distance(point1, point3))<= sqrt(distance(point1, point2)) + sqrt(distance(point2, point3)), "Triangle inequality test failed" 

    # test invalid inputs
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



def test_knn_constructor():
    """
    Checking that the constructor works properly
    """
    knn = kNN(1)
    knn = kNN(3)
    knn = kNN(1000)
    with pytest.raises(TypeError):
        knn = kNN('banana')
        # if the catched error is of the same kind of the one I expect, the test is passed
