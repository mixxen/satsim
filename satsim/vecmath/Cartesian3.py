import math

from .backend import xp


class Cartesian3:
    """A 3D Cartesian point backed by a ``numpy``/``cupy`` array."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """Constructor.

        Parameters
        ----------
        x, y, z : float
            Cartesian components.
        """
        self.v = xp.asarray([x, y, z], dtype=float)

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

    @property
    def z(self):
        return self.v[2]

    @z.setter
    def z(self, value):
        self.v[2] = value

    def __str__(self):
        return str([self.x, self.y, self.z])

    def __eq__(self, other):
        return (
            self is other or
            (self.x == other.x and
                self.y == other.y and
                self.z == other.z)
        )

    # ------------------------------------------------------------------
    # Convenience helpers using numpy/cupy
    # ------------------------------------------------------------------
    def to_array(self):
        """Return the underlying array without copying."""
        return self.v

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other):
        if isinstance(other, Cartesian3):
            return Cartesian3(self.x + other.x, self.y + other.y, self.z + other.z)
        arr = self.to_array() + other
        return Cartesian3.from_array(arr)

    def __sub__(self, other):
        if isinstance(other, Cartesian3):
            return Cartesian3(self.x - other.x, self.y - other.y, self.z - other.z)
        arr = self.to_array() - other
        return Cartesian3.from_array(arr)

    def __mul__(self, scalar):
        arr = self.to_array() * scalar
        return Cartesian3.from_array(arr)

    def __truediv__(self, scalar):
        arr = self.to_array() / scalar
        return Cartesian3.from_array(arr)

    def __neg__(self):
        arr = -self.to_array()
        return Cartesian3.from_array(arr)

    @staticmethod
    def fromSpherical(spherical, result=None):
        """ Converts the provided Spherical into Cartesian3 coordinates.

        Args:
            spherical: `dict`, The Spherical to be converted to Cartesian3.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter or a new Cartesian3 instance if one was not provided.
        """
        if result is None:
            result = Cartesian3()

        clock = spherical['clock']
        cone = spherical['cone']
        magnitude = spherical['magnitude']
        radial = magnitude * math.sin(cone)
        result.x = radial * math.cos(clock)
        result.y = radial * math.sin(clock)
        result.z = magnitude * math.cos(cone)
        return result

    @staticmethod
    def from_elements(x, y, z, result=None):
        """ Creates a Cartesian3 instance from x and y coordinates.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
            z: `float`, The Z component. default: 0
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter or a new Cartesian3 instance if one was not provided.
        """
        if result is None:
            return Cartesian3(x, y, z)

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def from_array(array, starting_index=0, result=None):
        """ Creates a Cartesian3 from three consecutive elements in an array.

        Args:
            array: `list`, The array whose three consecutive elements correspond to the x, y, and z components, respectively.
            starting_index: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter or a new Cartesian3 instance if one was not provided.
        """
        if result is None:
            result = Cartesian3()

        result.x = array[starting_index]
        result.y = array[starting_index + 1]
        result.z = array[starting_index + 2]
        return result

    @staticmethod
    def clone(cartesian=None, result=None):
        """ Duplicates a Cartesian3 instance.

        Args:
            cartesian: `Cartesian3`, The cartesian to duplicate.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter or a new Cartesian3 instance if one was not provided.
        """
        if cartesian is None:
            return None

        if result is None:
            return Cartesian3(cartesian.x, cartesian.y, cartesian.z)

        result.x = cartesian.x
        result.y = cartesian.y
        result.z = cartesian.z
        return result

    @staticmethod
    def maximum_component(cartesian):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian3`, The cartesian to use.

        Returns:
            A `number`, The value of the maximum component.
        """
        return max(cartesian.x, cartesian.y, cartesian.z)

    @staticmethod
    def minimum_component(cartesian):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian3`, The cartesian to use.

        Returns:
            A `number`, The value of the minimum component.
        """
        return min(cartesian.x, cartesian.y, cartesian.z)

    @staticmethod
    def minimum_by_component(first, second, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            first: `Cartesian3`, The cartesian to compare.
            second: `Cartesian3`, The cartesian to compare.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `number`, The value of the minimum component.
        """
        result.x = min(first.x, second.x)
        result.y = min(first.y, second.y)
        result.z = min(first.z, second.z)
        return result

    @staticmethod
    def maximum_by_component(first, second, result):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            first: `Cartesian3`, The cartesian to compare.
            second: `Cartesian3`, The cartesian to compare.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `number`, The value of the maximum component.
        """
        result.x = max(first.x, second.x)
        result.y = max(first.y, second.y)
        result.z = max(first.z, second.z)
        return result

    @staticmethod
    def clamp(value, min_val, max_val, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            value: `Cartesian3`, The value to clamp.
            min_val: `Cartesian3`, The minimum bound.
            max_val: `Cartesian3`, The maximum bound.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `number`, The clamped value such that min <= result <= max.
        """
        x = max(min(value.x, max_val.x), min_val.x)
        y = max(min(value.y, max_val.y), min_val.y)
        z = max(min(value.z, max_val.z), min_val.z)

        result.x = x
        result.y = y
        result.z = z

        return result

    @staticmethod
    def fromCartesian4(cartesian):
        """ Creates a Cartesian3 instance from an existing Cartesian3.  This simply takes the
        x and y properties of the Cartesian3 and drops z and w.

        Args:
            cartesian: `Cartesian4`, The cartesian to create from.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter or a new Cartesian3 instance if one was not provided.
        """
        return Cartesian3.clone(cartesian)

    @staticmethod
    def magnitude_squared(cartesian):
        """ Computes the provided Cartesian's squared magnitude.

        Args:
            cartesian: `Cartesian3`, he Cartesian instance whose squared magnitude is to be computed.

        Returns:
            A `float`, The squared magnitude.
        """
        return cartesian.x * cartesian.x + cartesian.y * cartesian.y + cartesian.z * cartesian.z

    @staticmethod
    def magnitude(cartesian):
        """ Computes the provided Cartesian's magnitude (length).

        Args:
            cartesian: `Cartesian3`, he Cartesian instance whose magnitude is to be computed.

        Returns:
            A `float`, The magnitude.
        """
        return math.sqrt(Cartesian3.magnitude_squared(cartesian))

    @staticmethod
    def distance(left, right):
        """ Computes the distance between two points.

        Args:
            left, `Cartesian3`, The first point to compute the distance from.
            right, `Cartesian3`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian3.subtract(left, right, _distanceScratch)
        return Cartesian3.magnitude(_distanceScratch)

    @staticmethod
    def distance_squared(left, right):
        """ Computes the squared distance between two points.  Comparing squared distances
        using this function is more efficient than comparing distances.

        Args:
            left, `Cartesian3`, The first point to compute the distance from.
            right, `Cartesian3`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian3.subtract(left, right, _distanceScratch)
        return Cartesian3.magnitude_squared(_distanceScratch)

    @staticmethod
    def normalize(cartesian, result):
        """ Computes the normalized form of the supplied Cartesian.

        Args:
            cartesian, `Cartesian3`, The Cartesian to be normalized.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        magnitude = Cartesian3.magnitude(cartesian)
        result.x = cartesian.x / magnitude
        result.y = cartesian.y / magnitude
        result.z = cartesian.z / magnitude
        return result

    @staticmethod
    def dot(left, right):
        """ Computes the dot (scalar) product of two Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.

        Returns:
            A `float`, The dot product.
        """
        return left.x * right.x + left.y * right.y + left.z * right.z

    @staticmethod
    def cross(left, right, result):
        """ Computes the magnitude of the cross product that would result from implicitly setting the Z coordinate of the input vectors to 0.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.

        Returns:
            A `float`, The cross product.
        """
        leftX = left.x
        leftY = left.y
        leftZ = left.z
        rightX = right.x
        rightY = right.y
        rightZ = right.z

        x = leftY * rightZ - leftZ * rightY
        y = leftZ * rightX - leftX * rightZ
        z = leftX * rightY - leftY * rightX

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def multiply_components(left, right, result):
        """ Computes the componentwise product of two Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = left.x * right.x
        result.y = left.y * right.y
        result.z = left.z * right.z
        return result

    @staticmethod
    def divide_components(left, right, result):
        """ Computes the component wise quotient of two Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x / right.x
        result.y = left.y / right.y
        result.z = left.z / right.z
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the component wise sum of two Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x + right.x
        result.y = left.y + right.y
        result.z = left.z + right.z
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the component wise difference of two Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = left.x - right.x
        result.y = left.y - right.y
        result.z = left.z - right.z
        return result

    @staticmethod
    def multiply_by_scalar(cartesian, scalar, result):
        """ Multiplies the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian3`, The Cartesian to be scaled.
            scalar, `float`, The scalar to multiply with.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = cartesian.x * scalar
        result.y = cartesian.y * scalar
        result.z = cartesian.z * scalar
        return result

    @staticmethod
    def divide_by_scalar(cartesian, scalar, result):
        """ Divides the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian3`, The Cartesian to be divided.
            scalar, `float`, The scalar to divide by.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = cartesian.x / scalar
        result.y = cartesian.y / scalar
        result.z = cartesian.z / scalar
        return result

    @staticmethod
    def negate(cartesian, result):
        """ Negates the provided Cartesian.

        Args:
            cartesian, `Cartesian3`, The Cartesian to be negated.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = -cartesian.x
        result.y = -cartesian.y
        result.z = -cartesian.z
        return result

    @staticmethod
    def abs(cartesian, result):
        """ Computes the absolute value of the provided Cartesian.

        Args:
            cartesian, `Cartesian3`, The Cartesian whose absolute value is to be computed.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = abs(cartesian.x)
        result.y = abs(cartesian.y)
        result.z = abs(cartesian.z)
        return result

    @staticmethod
    def lerp(start, end, t, result):
        """ Computes the linear interpolation or extrapolation at t using the provided cartesians.

        Args:
            start, `Cartesian3`, The value corresponding to t at 0.0.
            end, `Cartesian3`, The value corresponding to t at 1.0.
            t, `Cartesian3`, The point along t at which to interpolate.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        Cartesian3.multiply_by_scalar(end, t, _lerpScratch)
        result = Cartesian3.multiply_by_scalar(start, 1.0 - t, result)
        return Cartesian3.add(_lerpScratch, result, result)

    @staticmethod
    def angle_between(left, right):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right, `Cartesian3`, The second Cartesian.

        Returns:
            A `float`, The angle between the Cartesians.
        """
        Cartesian3.normalize(left, _angleBetweenScratch)
        Cartesian3.normalize(right, _angleBetweenScratch2)
        cosine = Cartesian3.dot(_angleBetweenScratch, _angleBetweenScratch2)
        sine = Cartesian3.magnitude(
            Cartesian3.cross(
                _angleBetweenScratch,
                _angleBetweenScratch2,
                _angleBetweenScratch
            )
        )
        return math.atan2(sine, cosine)

    @staticmethod
    def most_orthogonal_axis(cartesian, result):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            cartesian, `Cartesian3`, The Cartesian on which to find the most orthogonal axis.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The most orthogonal axis.
        """
        f = Cartesian3.normalize(cartesian, _mostOrthogonalAxisScratch)
        Cartesian3.abs(f, f)

        if (f.x <= f.y):
            if (f.x <= f.z):
                result = Cartesian3.clone(Cartesian3.UNIT_X(), result)
            else:
                result = Cartesian3.clone(Cartesian3.UNIT_Z(), result)
        elif f.y <= f.z:
            result = Cartesian3.clone(Cartesian3.UNIT_Y(), result)
        else:
            result = Cartesian3.clone(Cartesian3.UNIT_Z(), result)

        return result

    @staticmethod
    def projectVector(a, b, result):
        """ Projects vector a onto vector b.

        Args:
            a, `Cartesian3`, The vector that needs projecting.
            b: `Cartesian3`, The vector to project onto.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        scalar = Cartesian3.dot(a, b) / Cartesian3.dot(b, b)
        return Cartesian3.multiply_by_scalar(b, scalar, result)

    @staticmethod
    def equals(left, right):
        """ Compares the provided Cartesians componentwise and return `True` if equal,
        `False` otherwise

        Args:
            left, `Cartesian3`, The first Cartesian.
            right: `Cartesian3`, The second Cartesian.

        Returns:
            A `boolean`, `True` if equal, `False` otherwise.
        """
        return bool(
            left is right or
            (left is not None and right is not None and
                left.x == right.x and left.y == right.y and left.z == right.z)
        )

    @staticmethod
    def equals_epsilon(left, right, relative_epsilon=0.0, absolute_epsilon=0.0):
        """ Compares the provided Cartesians componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right: `Cartesian3`, The second Cartesian.
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
                math.isclose(left.y, right.y, rel_tol=relative_epsilon, abs_tol=absolute_epsilon) and
                math.isclose(left.z, right.z, rel_tol=relative_epsilon, abs_tol=absolute_epsilon))
        )

    @staticmethod
    def midpoint(left, right, result):
        """ Computes the midpoint between the right and left Cartesian.

        Args:
            left, `Cartesian3`, The first Cartesian.
            right: `Cartesian3`, The second Cartesian.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The midpoint.
        """
        result.x = (left.x + right.x) * 0.5
        result.y = (left.y + right.y) * 0.5
        result.z = (left.z + right.z) * 0.5

        return result

    @staticmethod
    def ZERO():
        """ A Cartesian3 instance initialized to (0.0, 0.0, 0.0)."""
        return Cartesian3(0,0,0)

    @staticmethod
    def ONE():
        """ A Cartesian3 instance initialized to (1.0, 1.0, 1.0)."""
        return Cartesian3(1,1,1)

    @staticmethod
    def UNIT_X():
        """ A Cartesian3 instance initialized to (1.0, 0.0, 0.0)."""
        return Cartesian3(1,0,0)

    @staticmethod
    def UNIT_Y():
        """ A Cartesian3 instance initialized to (0.0, 1.0, 0.0)."""
        return Cartesian3(0,1,0)

    @staticmethod
    def UNIT_Z():
        """ A Cartesian3 instance initialized to (0.0, 0.0, 1.0)."""
        return Cartesian3(0,0,1)


_lerpScratch = Cartesian3()
_angleBetweenScratch = Cartesian3()
_angleBetweenScratch2 = Cartesian3()
_mostOrthogonalAxisScratch = Cartesian3()
_distanceScratch = Cartesian3()

# Backwards compatibility aliases
fromElements = Cartesian3.from_elements
fromArray = Cartesian3.from_array
maximumComponent = Cartesian3.maximum_component
minimumComponent = Cartesian3.minimum_component
minimumByComponent = Cartesian3.minimum_by_component
maximumByComponent = Cartesian3.maximum_by_component
magnitudeSquared = Cartesian3.magnitude_squared
distanceSquared = Cartesian3.distance_squared
multiplyComponents = Cartesian3.multiply_components
divideComponents = Cartesian3.divide_components
multiplyByScalar = Cartesian3.multiply_by_scalar
divideByScalar = Cartesian3.divide_by_scalar
angleBetween = Cartesian3.angle_between
mostOrthogonalAxis = Cartesian3.most_orthogonal_axis
equalsEpsilon = Cartesian3.equals_epsilon

