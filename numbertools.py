from collections.abc import Iterable
from hashlib import sha512 as sha2_512
from hashlib import sha3_512
from math import ceil as CEIL
from math import floor as FLOOR
from math import gcd
from operator import index
from typing import Any, SupportsIndex
try:  # For better compatibility between Python versions
    from math import isqrt

    def Sqrt(n: SupportsIndex) -> int:
        """Return the largest integer whose square is at most n.

        Keyword arguments:\n
        n -- the radicand, a nonnegative integer-like object
        """
        N: int = index(n)
        if N < 0:
            raise ValueError
        return (isqrt(N))
except ImportError:
    def Sqrt(n: SupportsIndex) -> int:
        """Return the largest integer whose square is at most n.

        Keyword arguments:\n
        n -- the radicand, a nonnegative integer-like object
        """
        N: int = index(n)
        if N < 0:
            raise ValueError
        res: int = 1 << ((N.bit_length() + 1) >> 1)
        while True:
            new: int = (res + N//res) >> 1
            if new == res:
                break
            res = new
        if N//res < res:
            return (res - 1)
        return (res)

from gmpy2 import ceil as Ceil
from gmpy2 import floor as Floor
from gmpy2 import is_prime as mr  # Only used as a last-resort case
#                                    because it is relatively slow and
#                                    has been found multiple times in
#                                    the past to claim "primality" of
#                                    composite numbers
#                                   (Albrecht (2018) and Arnault (1995)
#                                    give excellent examples that have
#                                    fooled older versions)
from gmpy2 import is_strong_bpsw_prp as bpsw
from gmpy2 import is_strong_prp as miller_rabin
from gmpy2 import log as ln
from gmpy2 import mpz  # Fast computations for big integers
from gmpy2 import mpfr  # Arbitrary-precision real numbers
import gmpy2
from mpmath import ceil as ceiL
from mpmath import ceil as flooR
from mpmath import li
from numpy import ceil as ceiling
from numpy import floor as flooring
from numpy import nextafter
from numpy.random import Generator, SeedSequence
try:  # For better compatibility between numpy versions
    from numpy.random import PCG64DXSM  # if it exists, get it
    BITGENERATOR = PCG64DXSM
except ImportError:
    from numpy.random import PCG64  # otherwise, grab the normal version
    BITGENERATOR = PCG64

ctx = gmpy2.context()
ctx.precision = 65535
ctx.real_prec = 65535
ctx.imag_prec = 65535
ctx.round = gmpy2.RoundToNearest
ctx.real_round = gmpy2.RoundToNearest
ctx.imag_round = gmpy2.RoundToNearest
ctx.emax = 1073741823
ctx.emin = -1073741823
ctx.subnormalize = True
ctx.trap_underflow = True
ctx.trap_overflow = True
ctx.trap_inexact = False
ctx.trap_erange = True
ctx.trap_divzero = True
ctx.allow_complex = True
ctx.rational_division = True
ctx.allow_release_gil = False
gmpy2.set_context(ctx)


def floatify(x: Any
             ) -> float:
    """Return the nearest floating-point number to x.
    If there is more than one, return the one closer to 0.
    Returns 0 for NaNs

    Keyword arguments:\n
    n -- a number-like object to convert to a Python float
    """
    X: mpfr = mpfr(x)
    possible: list[float] = [float(X)]
    possible = [float(nextafter(possible[0], possible[0] - 1)),
                *possible,
                float(nextafter(possible[-1], possible[-1] + 1))]
    possible = [float(nextafter(possible[0], possible[0] - 1)),
                *possible,
                float(nextafter(possible[-1], possible[-1] + 1))]
    ranking: list[tuple[mpfr, float, float]] = [(-abs(mpfr(p) - X),
                                                 -abs(p),
                                                 p)
                                                for p in possible]
    try:
        return (max(ranking)[2])
    except gmpy2.RangeError:
        return (0)
    except BaseException:
        raise


def sha2(n: SupportsIndex
         ) -> int:
    """Return an integer representation of the SHA-2 512-bit hash of n.

    Keyword arguments:\n
    n -- an integer-like object
    """
    N: int = index(n)
    return (int.from_bytes(sha2_512(N.to_bytes((N.bit_length() + 7)//8,
                                               byteorder="big",
                                               signed=True)).digest(),
                           byteorder="big",
                           signed=False))


def sha3(n: SupportsIndex
         ) -> int:
    """Return an integer representation of the SHA-3 512-bit hash of n.

    Keyword arguments:\n
    n -- an integer-like object
    """
    N: int = index(n)
    return (int.from_bytes(sha3_512(N.to_bytes((N.bit_length() + 7)//8,
                                               byteorder="big",
                                               signed=True)).digest(),
                           byteorder="big",
                           signed=False))


def prng(seed: SupportsIndex
         ) -> Generator:
    """Return a PCG-64 (DXSM) PRNG with a 256-bit entropy pool.

    Keyword arguments:\n
    seed -- the seed to base the entropy on, an integer-like object
    """
    SEED: int = index(seed)
    SHA2: str = bin(sha2(SEED))[2:]
    SHA3: str = bin(sha3(SEED))[2:]
    ENTROPY: int = int(''.join([a + b
                                for a, b in zip(SHA2, SHA3)]),
                       base=2)
    return (Generator(BITGENERATOR(SeedSequence(entropy=ENTROPY,
                                                pool_size=8))))


def ceil(x: Any
         ) -> int:
    """Return the smallest integer greater than or equal to x.

    Supports types provided by:
     * standard Python
     * numpy
     * gmpy2
     * mpmath

    Other types may be compatible, but this behavior is not guaranteed.

    Keyword arguments:\n
    x -- a real number-like object
    """
    try:
        return (int(CEIL(x)))
    except (TypeError, ValueError, OverflowError):
        try:
            return (int(ceiling(x)))
        except (TypeError, ValueError, OverflowError):
            try:
                return (int(Ceil(x)))
            except (TypeError, ValueError, OverflowError):
                try:
                    return (int(ceiL(x)))
                except (TypeError, ValueError, OverflowError):
                    pass
    raise


def floor(x: Any
          ) -> int:
    """Return the largest integer less than or equal to x.

    Supports types provided by:
     * standard Python
     * numpy
     * gmpy2
     * mpmath

    Other types may be compatible, but this behavior is not guaranteed.

    Keyword arguments:\n
    x -- a real number-like object
    """
    try:
        return (int(FLOOR(x)))
    except (TypeError, ValueError, OverflowError):
        try:
            return (int(flooring(x)))
        except (TypeError, ValueError, OverflowError):
            try:
                return (int(Floor(x)))
            except (TypeError, ValueError, OverflowError):
                try:
                    return (int(flooR(x)))
                except (TypeError, ValueError, OverflowError):
                    pass
    raise


def ceildiv(a: SupportsIndex,
            b: SupportsIndex
            ) -> int:
    """Return the smallest integer n such that A is not more than B*n.

    Keyword arguments:\n
    a -- the dividend, an integer-like object\n
    b -- the divisor, a nonzero integer-like object
    """
    A: int = index(a)
    B: int = index(b)
    if B * (A // B) == A:
        return (A // B)
    return ((A // B) + 1)


def Sieve(n: SupportsIndex
          ) -> tuple[list[int], int]:
    """A list-based quasi-wheeled prime sieve with skip interval 2.

    Return a tuple containing:
     * a list of all primes less than or equal to n
     * the product of all primes less than or equal to n

    Keyword arguments:\n
    n -- an integer-like object greater than or equal to 2
    """
    N: int = index(n)
    if N < 2:
        raise ValueError
    elif N < 3:
        return (([2], 2))
    elif N < 5:
        return (([2, 3], 6))
    elif N < 7:
        return (([2, 3, 5], 30))
    nvals: int = (N - 1)//2
    primes: list[bool] = [True]*nvals
    sN: int = (Sqrt(n) + 1)//2
    for i in range(sN):
        if primes[i]:
            for f in range((((2 * i) + 3)**2 - 3)//2, nvals, 2*i + 3):
                primes[f] = False
    lst: list[int] = [2]
    prd: mpz = mpz(2)
    for i in range(nvals):
        if primes[i]:
            lst.append(2*i + 3)
            prd = prd * (2*i + 3)
    return ((lst, int(prd)))


# A 6543-element set of sample primes, and their product
prelim_primes, prod_prelim_primes = Sieve(65537)


def is_prime(n: SupportsIndex
             ) -> bool:
    """Provide a probabilistic assessment of the primality of N.

    A combination of a divisibility test for a set of sample primes,
        a 36-iteration Miller-Rabin test, and a strong BPSW test.

    Guaranteed correct for all values up to 3317044064679887385961981
        (which includes all integers up to 81 bits).

    Probability of inaccuracy for arbitrary values is well below 4^-36.

    Keyword arguments:\n
    n -- an integer-like object greater than or equal to 2
    """
    N: int = index(n)
    if N < 2:
        return (False)
    if gcd(N, prod_prelim_primes) == 1:
        if N < (prelim_primes[-1] + 2)**2:
            return (True)
        if N < (1 << 64):
            return (bpsw(N))
        if N < 0x2be6951adc5b22410a5fd:
            for a in prelim_primes[:13]:
                if miller_rabin(a, N):
                    return (False)
            return (True)
        if bpsw(N):
            return (mr(N, 36))
        return (False)
    return (False)


def antiprimorial(n: SupportsIndex
                  ) -> tuple[int, int]:
    """Return a tuple consisting of the smallest primorial greater than
        or equal to n, as well as what it is the primorial of.

    Keyword arguments:\n
    n -- an integer-like object greater than or equal to 1
    """
    N: int = index(n)
    if N < 1:
        raise ValueError
    if N < 3:
        return ((2, 2))
    if N < 7:
        return ((6, 3))
    if N < 31:
        return ((30, 5))
    if N < 211:
        return ((210, 7))
    if N < 2311:
        return ((2310, 11))
    if N < 30031:
        return ((30030, 13))
    if N < 510511:
        return ((510510, 17))
    consider: list[int] = Sieve(N.bit_length())[0]
    mindex: int = 7
    primorial: mpz = mpz(9699690)
    while primorial < N:
        mindex += 1
        primorial *= consider[mindex]
    return ((int(primorial), consider[mindex]))


def between(low: SupportsIndex,
            high: SupportsIndex
            ) -> int:
    """Return an upper-bound estimate for the number of primes
        between low and high

    Keyword arguments:\n
    low -- the lower bound on the interval, an integer-like object\n
    high -- the upper bound on the interval, an integer-like object
    """
    L: int = index(low)
    H: int = index(high)
    # According to my comments on an old project, this is a decently
    # # close approximation to, and a guaranteed upper bound on, the
    # # number of primes between low and high for high > low > 620
    # Unfortunately, I cannot recall where I got it from. It seems to
    # # work, but I have no idea why.
    return (ceil((H / ln(H)) * 1.105) - floor(li(L, offset=True) * 0.89))


def number_to_base(n: SupportsIndex,
                   b: SupportsIndex
                   ) -> list[int]:
    """Return a list of int constituting a base-b representation of n.

    Keyword arguments:\n
    n -- the value to be represented, a nonnegative integer-like object\n
    b -- the radix, an integer-like object greater than or equal to 2
    """
    number: int = index(n)
    if number < 0:
        raise ValueError
    radix: int = index(b)
    if radix < 2:
        raise ValueError
    if number == 0:
        return ([0])
    digits: list[int] = []
    while number > 0:
        digits.append(int(number % radix))
        number = number // radix
    return (digits[::-1])


def base_to_number(d: Iterable[SupportsIndex],
                   b: SupportsIndex
                   ) -> int:
    """Return an integer from a base-b representation of that integer.

    Keyword arguments:\n
    d -- the digits (in big-endian order) of a base-b numeral,
            an iterable of nonnegative integer-like objects
    b -- the radix, an integer-like object greater than or equal to 2
    """
    digits: list[int] = [index(n) for n in d]
    radix: int = index(b)
    if radix < 2:
        raise ValueError
    res: int = 0
    for i, digit in enumerate(digits[::-1]):
        if digit < 0:
            raise ValueError
        else:
            res += digit * radix**i
    return (res)
