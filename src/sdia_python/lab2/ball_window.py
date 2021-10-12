import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BallWindow:
    """Creates a window of ball shape"""

    def __init__(self, center, radius):
        """Initialize the ball window with the center point and the radius .

        Args:
            center (array): array of the center point
            radius (float): size of the radius of the ball window
        """
        self.center = center
        self.radius = radius

    def dimension(self):
        """Returns the dimension of the ball window

        Returns:
            integer : dimension of the ball window
        """
        return self.center.size

    def volume(self):
        """Returns the volume created by the ball window

        Returns:
            integer : volume of the ball
        """
        dim = self.dimension()
        if dim == 1:
            return 2 * self.radius
        if dim == 2:
            return np.pi * self.radius ** 2
        if dim == 3:
            return (4 / 3) * np.pi * self.radius ** 3
        raise Exception("dimension too high")

    def __contains__(self, point):
        """Indicates if the point is contained in the ball window.
        Returns True if the point is in the ball, returns False otherwise.

        Args:
            point (array): coordinates of the point that we want to know if it is part of the ball.
        """
        if point.shape != self.center.shape:
            raise ValueError(
                "the dimension of the point should be the same as the dimension of the center"
            )
        N = np.linalg.norm(point - self.center)
        return np.all(N <= self.radius)

    def indicator_function(self, point):
        """Indicator function of the ball window. Returns 1 if the point is in the ball, returns 0 otherwise.

        Args:
            point (array): coordinates of the point
        """
        return 1 if point in self else 0

    def rand(self, n=1, rng=None):
        """Generate ``n`` points uniformly at random inside the :py:class:`BallWindow`.

        Args:
            n (int, optional): [description]. Defaults to 1.
            rng ([type], optional): [description]. Defaults to None.
        """
        rng = get_random_number_generator(rng)
        L = []
        while len(L) < n:  # nb of points taken randomly in the box
            point = rng.uniform(self.center - self.radius, self.center + self.radius)
            if np.linalg.norm(self.center - point) <= self.radius:
                L.append(point)
        return np.array(L)


class UnitBallWindow(BallWindow):
    def __init__(self, center):
        """Initialize unitary ball window given an array for the center point(s) and the radius will be 1.

        Args:
            center (np.array()): represents the point of the center of the ball window. Its shape is (1,n) with n the dimension of the unitary ball window
            Defaults to None.
        """
        self.center = center
        super(UnitBallWindow, self).__init__(center, 1)
