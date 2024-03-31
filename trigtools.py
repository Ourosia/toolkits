import math
from typing import SupportsFloat
import numpy as np
import numpy.typing as npt


def periodic_normalize(x: npt.NDArray[np.floating] | SupportsFloat
                       ) -> npt.NDArray[np.float64] | float:
    """Find the unique value(s) v in [-1, 1) such that
    f(v) = f(x) for any function f of period 2.

    Keyword arguments:\n
    x -- A value or array of values to wrap to the range [-1, 1).
    """
    if isinstance(x, np.ndarray):
        # cast to float64
        X_arr: npt.NDArray[np.float64] = np.array(x, dtype=np.float64)
        # get sign in compatible format
        X_sgn: npt.NDArray[np.float64] = np.sign(X_arr).astype(np.float64)
        # shift by sign (toward 0 to prevent reducing accuracy)
        X_arr -= X_sgn
        # trim to [0, 2)
        X_arr %= 2
        # shift back, placing us in the range [-1, 1)
        X_arr -= X_sgn**2
        return (X_arr)
    else:
        # cast to float
        X_val: float = float(x)
        # get sign in compatible format
        sgn_X: float = math.copysign(1.0, X_val)
        # shift by sign (toward 0 to prevent reducing accuracy)
        X_val -= sgn_X
        # trim to [0, 2)
        X_val %= 2
        # shift back, placing us in the range [-1, 1)
        X_val -= sgn_X**2
        return (X_val)


def sin_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the sine of pi * x."""
    if isinstance(x, np.ndarray):
        return (np.sin(np.pi * periodic_normalize(x)).astype(np.float64))
    else:
        return (math.sin(math.pi * periodic_normalize(x)))


def array_sin_pi(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Compute the sine of pi * x, with x an array of floats."""
    return (np.array(sin_pi(x), dtype=np.float64))


def scalar_sin_pi(x: SupportsFloat) -> float:
    """Compute the sine of pi * x, with x a floating-point number."""
    return (float(sin_pi(x)))


def cos_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the cosine of pi * x."""
    if isinstance(x, np.ndarray):
        return (np.cos(np.pi * periodic_normalize(x)).astype(np.float64))
    else:
        return (math.cos(math.pi * periodic_normalize(x)))


def array_cos_pi(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Compute the cosine of pi * x, with x an array of floats."""
    return (np.array(cos_pi(x), dtype=np.float64))


def scalar_cos_pi(x: SupportsFloat) -> float:
    """Compute the cosine of pi * x, with x a floating-point number."""
    return (float(cos_pi(x)))


def tan_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the tangent of pi * x."""
    if isinstance(x, np.ndarray):
        return (np.tan(np.pi * periodic_normalize(x)).astype(np.float64))
    else:
        return (math.tan(math.pi * periodic_normalize(x)))


def cot_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the cotangent of pi * x."""
    return (1/tan_pi(x))


def sec_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the secant of pi * x."""
    return (1/cos_pi(x))


def csc_pi(x: npt.NDArray[np.floating] | SupportsFloat
           ) -> npt.NDArray[np.float64] | float:
    """Compute the cosecant of pi * x."""
    return (1/sin_pi(x))
