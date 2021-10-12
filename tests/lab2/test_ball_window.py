import numpy as np
import pytest

from sdia_python.lab2.ball_window import BallWindow, UnitBallWindow

# checks if the ball window created has the correct radius
@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([0]), 4, 4),
        (np.array([2.5, 2.5]), 3.7, 3.7),
        (np.array([-1, 5]), 2.0, 2.0),
        (np.array([10, 3]), 20000, 20000),
    ],
)
def test_radius(center, radius, expected):
    ball = BallWindow(center, radius)
    assert ball.radius == expected


# checks if the ball window created has the correct dimension
@pytest.mark.parametrize(
    "box, expected",
    [
        (np.array([1]), 1),
        (np.array([1, 2.2]), 2),
        (np.array([1.4, 2.6, 3.9]), 3),
    ],
)
def test_dimension_box(box, expected):
    ball = BallWindow(box, 4)
    assert ball.dimension() == expected


# checks if the ball window created has the correct volume
@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([1]), 2, 4),
        (np.array([1, 3]), 3, 9 * np.pi),
        (np.array([1.4, 2.6, 3.9]), 2.5, (4 / 3) * np.pi * 2.5 ** 3),
    ],
)
def test_volume_box(center, radius, expected):
    ball = BallWindow(center, radius)
    assert ball.volume() == expected


# checks for dimension =1 if a point is contained in a ball window.
def test_contains_oneDimension():
    ball1 = BallWindow(np.array([1]), 3)
    ball2 = BallWindow(np.array([3.5]), 0.5)
    ball3 = BallWindow(np.array([-2.5]), 1.5)
    ball4 = BallWindow(np.array([0]), 2)
    assert ball1.__contains__(np.array([2]))
    assert not ball1.__contains__(np.array([5]))
    assert ball2.__contains__(np.array([4]))
    assert not ball2.__contains__(np.array([4.01]))
    assert ball3.__contains__(np.array([-3.9]))
    assert ball4.__contains__(np.array([0]))
    assert not ball4.__contains__(np.array([2.02]))


# checks for dimension =2 if a point is contained in a ball window.
def test_contains_twoDimension():
    ball1 = BallWindow(np.array([0, 0]), 1)
    ball2 = BallWindow(np.array([3.5, 2.5]), 0.5)
    assert ball1.__contains__(np.array([0.5, 0.5]))
    assert not ball1.__contains__(np.array([1, 2]))
    assert ball2.__contains__(np.array([3.25, 2.75]))


# checks the indicator function for dimension =1.
def test_indicator_function_oneDimension():
    ball1 = BallWindow(np.array([1]), 3)
    ball2 = BallWindow(np.array([3.5]), 0.5)
    ball3 = BallWindow(np.array([-2.5]), 1.5)
    ball4 = BallWindow(np.array([0]), 2)
    assert ball1.indicator_function(np.array([2]))
    assert not ball1.indicator_function(np.array([5]))
    assert ball2.indicator_function(np.array([4]))
    assert not ball2.indicator_function(np.array([4.01]))
    assert ball3.indicator_function(np.array([-3.9]))
    assert ball4.indicator_function(np.array([0]))
    assert not ball4.indicator_function(np.array([2.02]))


# checks the indicator function for dimension =2.
def test_indicator_function_twoDimension():
    ball1 = BallWindow(np.array([0, 0]), 1)
    ball2 = BallWindow(np.array([3.5, 2.5]), 0.5)
    assert ball1.indicator_function(np.array([0.5, 0.5]))
    assert not ball1.indicator_function(np.array([1, 2]))
    assert ball2.indicator_function(np.array([3.25, 2.75]))


# checks if the ValueError is raised when using the function __contains__
def test_raise_value_error_when_points_is_not_of_good_dimension():
    with pytest.raises(ValueError):
        ball1 = BallWindow(np.array([0, 0]), 1)
        np.array([1, 2, 3]) in ball1


# checks if 100 randomly taken points with rand are in the ball window as they should be.
def test_rand_multiplepoint_3dimension():
    ball = BallWindow(np.array([1, 15.5, 3.5]), 2)
    coord = ball.rand(100)
    for value in coord:
        assert ball.__contains__(value)


# checks if the ball window created is unitary (the length of radius = 1)
@pytest.mark.parametrize(
    "center, expected",
    [
        (np.array([2, 3]), 1),
        (np.array([1, 1, 1]), 1),
        (np.array([0]), 1),
    ],
)
def test_UnitBallWindow_init(center, expected):
    d = UnitBallWindow(center)
    assert np.all(d.radius == expected)
