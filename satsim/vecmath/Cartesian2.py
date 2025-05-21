import math

from .backend import xp


class Cartesian2:
    """A 2D Cartesian point backed by a :mod:`numpy`/``cupy`` array."""

    def __init__(self, x=0.0, y=0.0):
        """Constructor.

        Parameters
        ----------
        x : float
            The X component.
        y : float
            The Y component.
        """
        self.v = xp.asarray([x, y], dtype=float)

    # ------------------------------------------------------------------
    # Properties mapping to the underlying array
    # ------------------------------------------------------------------
    @property
    def x(self):
        return self.v[0]

    @x.setter
    def x(self, value):
        self.v[0] = value

    @property
    def y(self):
        return self.v[1]

    @y.setter
    def y(self, value):
        self.v[1] = value

    def __str__(self):
        return str([self.x, self.y])

    def __eq__(self, other):
        return (
            self is other or
            (self.x == other.x and self.y == other.y)
        )

    # ------------------------------------------------------------------
    # Convenience helpers using numpy/cupy
    # ------------------------------------------------------------------
    def to_array(self):
        """Return the underlying array without copying."""
        return self.v

    @classmethod
    def from_array(cls, arr):
        """Create a :class:`Cartesian2` from a numpy/cupy array."""
        return cls(arr[0], arr[1])

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        if isinstance(other, Cartesian2):
            return Cartesian2(self.x + other.x, self.y + other.y)
        arr = self.to_array() + other
        return Cartesian2.from_array(arr)

    def __sub__(self, other):
        if isinstance(other, Cartesian2):
            return Cartesian2(self.x - other.x, self.y - other.y)
        arr = self.to_array() - other
        return Cartesian2.from_array(arr)

    def __mul__(self, scalar):
        arr = self.to_array() * scalar
        return Cartesian2.from_array(arr)

    def __truediv__(self, scalar):
        arr = self.to_array() / scalar
        return Cartesian2.from_array(arr)

    def __neg__(self):
        arr = -self.to_array()
        return Cartesian2.from_array(arr)

    @staticmethod
    def from_elements(x, y, result=None):
        """ Creates a Cartesian2 instance from x and y coordinates.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if result is None:
            return Cartesian2(x, y)

        result.x = x
        result.y = y
        return result

    @staticmethod
    def from_array(array, starting_index=0, result=None):
        """ Creates a Cartesian2 from two consecutive elements in an array.

        Args:
            array: `list`, The array whose two consecutive elements correspond to the x and y components, respectively.
            starting_index: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if result is None:
            result = Cartesian2()

        result.x = array[starting_index]
        result.y = array[starting_index + 1]
        return result

    @staticmethod
    def clone(cartesian=None, result=None):
        """ Duplicates a Cartesian2 instance.

        Args:
            cartesian: `Cartesian2`, The cartesian to duplicate.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if cartesian is None:
            return None

        if result is None:
            return Cartesian2(cartesian.x, cartesian.y)

        result.x = cartesian.x
        result.y = cartesian.y
        return result

    @staticmethod
    def maximum_component(cartesian):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian2`, The cartesian to use.

        Returns:
            A `number`, The value of the maximum component.
        """
        return max(cartesian.x, cartesian.y)

    @staticmethod
    def minimum_component(cartesian):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian2`, The cartesian to use.

        Returns:
            A `number`, The value of the minimum component.
        """
        return min(cartesian.x, cartesian.y)

    @staticmethod
    def minimum_by_component(first, second, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            first: `Cartesian2`, The cartesian to compare.
            second: `Cartesian2`, The cartesian to compare.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The value of the minimum component.
        """
        result.x = min(first.x, second.x)
        result.y = min(first.y, second.y)
        return result

    @staticmethod
    def maximum_by_component(first, second, result):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            first: `Cartesian2`, The cartesian to compare.
            second: `Cartesian2`, The cartesian to compare.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The value of the maximum component.
        """
        result.x = max(first.x, second.x)
        result.y = max(first.y, second.y)
        return result

    @staticmethod
    def clamp(value, min_val, max_val, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            value: `Cartesian2`, The value to clamp.
            min_val: `Cartesian2`, The minimum bound.
            max_val: `Cartesian2`, The maximum bound.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The clamped value such that min <= result <= max.
        """
        x = max(min(value.x, max_val.x), min_val.x)
        y = max(min(value.y, max_val.y), min_val.y)

        result.x = x
        result.y = y

        return result

    @staticmethod
    def from_cartesian3(cartesian, result=None):
        """ Creates a Cartesian2 instance from an existing Cartesian3.  This simply takes the
        x and y properties of the Cartesian3 and drops z.

        Args:
            cartesian: `Cartesian3`, The cartesian to create from.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        return Cartesian2.clone(cartesian, result)

    @staticmethod
    def from_cartesian4(cartesian, result=None):
        """ Creates a Cartesian2 instance from an existing Cartesian3.  This simply takes the
        x and y properties of the Cartesian3 and drops z and w.

        Args:
            cartesian: `Cartesian4`, The cartesian to create from.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        return Cartesian2.clone(cartesian, result=None)

    @staticmethod
    def magnitude_squared(cartesian):
        """ Computes the provided Cartesian's squared magnitude.

        Args:
            cartesian: `Cartesian2`, he Cartesian instance whose squared magnitude is to be computed.

        Returns:
            A `float`, The squared magnitude.
        """
        arr = cartesian.to_array()
        return xp.dot(arr, arr)

    @staticmethod
    def magnitude(cartesian):
        """ Computes the provided Cartesian's magnitude (length).

        Args:
            cartesian: `Cartesian2`, he Cartesian instance whose magnitude is to be computed.

        Returns:
            A `float`, The magnitude.
        """
        arr = cartesian.to_array()
        return xp.linalg.norm(arr)

    @staticmethod
    def distance(left, right):
        """ Computes the distance between two points.

        Args:
            left, `Cartesian2`, The first point to compute the distance from.
            right, `Cartesian2`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        diff = left.to_array() - right.to_array()
        return xp.linalg.norm(diff)

    @staticmethod
    def distance_squared(left, right):
        """ Computes the squared distance between two points.  Comparing squared distances
        using this function is more efficient than comparing distances.

        Args:
            left, `Cartesian2`, The first point to compute the distance from.
            right, `Cartesian2`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        diff = left.to_array() - right.to_array()
        return xp.dot(diff, diff)

    @staticmethod
    def normalize(cartesian, result):
        """ Computes the normalized form of the supplied Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be normalized.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = cartesian.to_array()
        magnitude = xp.linalg.norm(arr)
        result.x = arr[0] / magnitude
        result.y = arr[1] / magnitude
        return result

    @staticmethod
    def dot(left, right):
        """ Computes the dot (scalar) product of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The dot product.
        """
        return xp.dot(left.to_array(), right.to_array())

    @staticmethod
    def cross(left, right):
        """ Computes the magnitude of the cross product that would result from implicitly setting the Z coordinate of the input vectors to 0.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The cross product.
        """
        return left.x * right.y - left.y * right.x

    @staticmethod
    def multiply_components(left, right, result):
        """ Computes the componentwise product of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = left.to_array() * right.to_array()
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def divide_components(left, right, result):
        """ Computes the component wise quotient of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        arr = left.to_array() / right.to_array()
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the component wise sum of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        arr = left.to_array() + right.to_array()
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the component wise difference of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = left.to_array() - right.to_array()
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def multiply_by_scalar(cartesian, scalar, result):
        """ Multiplies the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be scaled.
            scalar, `float`, The scalar to multiply with.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = cartesian.to_array() * scalar
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def divide_by_scalar(cartesian, scalar, result):
        """ Divides the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be divided.
            scalar, `float`, The scalar to divide by.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = cartesian.to_array() / scalar
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def negate(cartesian, result):
        """ Negates the provided Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be negated.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = -cartesian.to_array()
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def abs(cartesian, result):
        """ Computes the absolute value of the provided Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian whose absolute value is to be computed.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        arr = xp.abs(cartesian.to_array())
        result.x = arr[0]
        result.y = arr[1]
        return result

    @staticmethod
    def lerp(start, end, t, result):
        """ Computes the linear interpolation or extrapolation at t using the provided cartesians.

        Args:
            start, `Cartesian2`, The value corresponding to t at 0.0.
            end, `Cartesian2`, The value corresponding to t at 1.0.
            t, `Cartesian2`, The point along t at which to interpolate.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        Cartesian2.multiply_by_scalar(end, t, _lerpScratch)
        result = Cartesian2.multiply_by_scalar(start, 1.0 - t, result)
        return Cartesian2.add(_lerpScratch, result, result)

    @staticmethod
    def angle_between(left, right):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The angle between the Cartesians.
        """
        Cartesian2.normalize(left, _angleBetweenScratch)
        Cartesian2.normalize(right, _angleBetweenScratch2)
        return math.acos(max(min(Cartesian2.dot(_angleBetweenScratch, _angleBetweenScratch2), 1.0), -1.0))

    @staticmethod
    def most_orthogonal_axis(cartesian, result):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            cartesian, `Cartesian2`, The Cartesian on which to find the most orthogonal axis.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The most orthogonal axis.
        """
        f = Cartesian2.normalize(cartesian, _mostOrthogonalAxisScratch)
        Cartesian2.abs(f, f)
        if f.x <= f.y:
            result = Cartesian2.clone(Cartesian2.UNIT_X(), result)
        else:
            result = Cartesian2.clone(Cartesian2.UNIT_Y(), result)
        return result

    @staticmethod
    def equals(left, right):
        """ Compares the provided Cartesians componentwise and return `True` if equal,
        `False` otherwise.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right: `Cartesian2`, The second Cartesian.

        Returns:
            A `boolean`, `True` if equal, `False` otherwise.
        """
        return bool(
            left is right or
            (left is not None and right is not None and
                left.x == right.x and left.y == right.y)
        )

    @staticmethod
    def equals_epsilon(left, right, relative_epsilon=0.0, absolute_epsilon=0.0):
        """ Compares the provided Cartesians componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right: `Cartesian2`, The second Cartesian.
            relative_epsilon, `float`, The relative epsilon tolerance to use for equality testing. default=0
            absolute_epsilon, `float`, The absolute epsilon tolerance to use for equality testing. default=0

        Returns:
            A `boolean`, `True` if they pass an absolute or relative tolerance test, `False` otherwise.
        """
        return bool(
            left is right or
            (left is not None and
                right is not None and
                math.isclose(left.x, right.x, rel_tol=relative_epsilon, abs_tol=absolute_epsilon) and
                math.isclose(left.y, right.y, rel_tol=relative_epsilon, abs_tol=absolute_epsilon))
        )

    @staticmethod
    def ZERO():
        """ A Cartesian2 instance initialized to (0.0, 0.0)."""
        return Cartesian2(0,0)

    @staticmethod
    def ONE():
        """ A Cartesian2 instance initialized to (1.0, 1.0)."""
        return Cartesian2(1,1)

    @staticmethod
    def UNIT_X():
        """ A Cartesian2 instance initialized to (1.0, 0.0)."""
        return Cartesian2(1,0)

    @staticmethod
    def UNIT_Y():
        """ A Cartesian2 instance initialized to (0.0, 1.0)."""
        return Cartesian2(0,1)


_angleBetweenScratch = Cartesian2()
_angleBetweenScratch2 = Cartesian2()
_lerpScratch = Cartesian2()
_mostOrthogonalAxisScratch = Cartesian2()
_distanceScratch = Cartesian2()

# Backwards compatibility aliases
fromElements = Cartesian2.from_elements
fromArray = Cartesian2.from_array
maximumComponent = Cartesian2.maximum_component
minimumComponent = Cartesian2.minimum_component
minimumByComponent = Cartesian2.minimum_by_component
maximumByComponent = Cartesian2.maximum_by_component
fromCartesian3 = Cartesian2.from_cartesian3
fromCartesian4 = Cartesian2.from_cartesian4
magnitudeSquared = Cartesian2.magnitude_squared
distanceSquared = Cartesian2.distance_squared
multiplyComponents = Cartesian2.multiply_components
divideComponents = Cartesian2.divide_components
multiplyByScalar = Cartesian2.multiply_by_scalar
divideByScalar = Cartesian2.divide_by_scalar
angleBetween = Cartesian2.angle_between
mostOrthogonalAxis = Cartesian2.most_orthogonal_axis
equalsEpsilon = Cartesian2.equals_epsilon

