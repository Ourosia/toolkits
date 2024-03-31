from collections.abc import Callable
from typing import Any, SupportsFloat


def ODE1_step(t0: SupportsFloat,
              x0: Any,
              f: Callable[[Any, Any], Any],
              dt: SupportsFloat,
              m: Callable[[Any], SupportsFloat],
              b: None | SupportsFloat = None
              ) -> tuple[float, Any]:
    """Run the unique 1st order method (Euler's) for an ODE
        x' = f(t, x),
    starting at t = t0 and x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

      0 |

     ---+---

        | 1
    """
    T0 = float(t0)
    DT = float(dt)
    dx = DT * f(T0, x0)
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = ODE1_step(T0, x0, f, DT/2, m, b)
            return (ODE1_step(t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))


def gen_ODE2_step(c1: SupportsFloat,
                  t0: SupportsFloat,
                  x0: Any,
                  f: Callable[[Any, Any], Any],
                  dt: SupportsFloat,
                  m: Callable[[Any], SupportsFloat],
                  b: None | SupportsFloat = None
                  ) -> tuple[float, Any]:
    """Run a generic 2-stage 2nd order method for an ODE
        x' = f(t, x),
    starting at t = t0 and x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

       0 |

      c1 | a11

     ----+-----------

         |  b1    b2

    c1 = c1 (keyword argument)

    a11 = c1

    b1 = (2 c1 - 1)/(2 c1)

    b2 = 1/(2 c1)
    """
    C1: float = float(c1)
    T0: float = float(t0)
    DT: float = float(dt)
    k1 = f(T0, x0)
    k2 = f(T0 + (C1 * DT),
           x0 + (DT * C1 * k1))
    dx = DT * ((((2 * C1) - 1)/(2 * C1)) * k1 +
               (1/(2 * C1)) * k2)
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = gen_ODE2_step(C1, T0, x0, f, DT/2, m, b)
            return (gen_ODE2_step(C1, t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))


def RL_ODE2_step(t0: SupportsFloat,
                 x0: Any,
                 f: Callable[[Any, Any], Any],
                 dt: SupportsFloat,
                 m: Callable[[Any], SupportsFloat],
                 b: None | SupportsFloat = None
                 ) -> tuple[float, Any]:
    """Run the optimal 2nd order method (under the Ralston/Lotkin error
    bound judgement scheme) for an ODE
        x' = f(t, x),
    starting at t = t0 and x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

      0 |

      a | a

     ---+-------

        | b   c

    a = 2/3
        (implemented as 6004799503160661/9007199254740992)

    b = 1/4

    c = 3/4
    """
    T0: float = float(t0)
    DT: float = float(dt)
    k1 = f(T0, x0)
    k2 = f(T0 + ((0x15555555555555/(1 << 53)) * DT),
           x0 + (DT * ((0x15555555555555/(1 << 53)) * k1)))
    dx = DT * ((1/4) * k1 +
               (3/4) * k2)
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = RL_ODE2_step(T0, x0, f, DT/2, m, b)
            return (RL_ODE2_step(t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))


def gen_ODE3_step(c1: SupportsFloat,
                  c2: SupportsFloat,
                  t0: SupportsFloat,
                  x0: Any,
                  f: Callable[[Any, Any], Any],
                  dt: SupportsFloat,
                  m: Callable[[Any], SupportsFloat],
                  b: None | SupportsFloat = None
                  ) -> tuple[float, Any]:
    """Run a generic 3-stage 3rd order method for an ODE
        x' = f(t, x),
    starting at t = t0 and x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

       0 |

      c1 | a11

      c2 | a21   a22

     ----+-----------------

         |  b1    b2    b3

    c1 = c1 (keyword argument)

    c2 = c2 (keyword argument)

    a11 = c1

    a21 = c2 - (c2^2)/(2 c1)

    a22 = (c2^2)/(2 c1)

    b1 = 1 - (2 c1 - 2 c2 + 3 c2^2)/(6 c1 c2^2)

    b2 = (3 c2 - 2)/(6 c1 c2)

    b3 = 1/(3 c2^2)
    """
    C1: float = float(c1)
    C2: float = float(c2)
    T0: float = float(t0)
    DT: float = float(dt)
    k1 = f(T0, x0)
    k2 = f(T0 + (C1 * DT),
           x0 + (DT * C1 * k1))
    k3 = f(T0 + (C2 * DT),
           x0 + (DT * ((C2 - (C2**2)/(2 * C1)) * k1 +
                       ((C2**2)/(2 * C1)) * k2)))
    dx = DT * ((1 - (2*C1 - 2*C2 + (3 * C2**2))/(6 * C1 * C2**2)) * k1 +
               (((3 * C2) - 2)/(6 * C1 * C2)) * k2 +
               (1/(2 * C2**2)) * k3)
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = gen_ODE3_step(C1, C2, T0, x0, f, DT/2, m, b)
            return (gen_ODE3_step(C1, C2, t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))


def RL_ODE3_step(t0: SupportsFloat,
                 x0: Any,
                 f: Callable[[Any, Any], Any],
                 dt: SupportsFloat,
                 m: Callable[[Any], SupportsFloat],
                 b: None | SupportsFloat = None
                 ) -> tuple[float, Any]:
    """Run the optimal 3rd order method (under the Ralston/Lotkin error
    bound judgement scheme) for an ODE
        x' = f(t, x),
    starting at t = t0, x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

      0 |

      a | a

      b | c   d

     ---+-----------

        | e   f   g

    a = 1/2

    b = 3/4

    c = 0

    d = 3/4

    e = 2/9

    f = 1/3
        (implemented as 3/9)

    g = 4/9
    """
    T0: float = float(t0)
    DT: float = float(dt)
    k1 = f(T0, x0)
    k2 = f(T0 + DT/2,
           x0 + (DT * (k1/2)))
    k3 = f(T0 + ((3/4) * DT),
           x0 + (DT * ((3/4) * k2)))
    dx = DT * (2 * k1 +
               3 * k2 +
               4 * k3)/9
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = RL_ODE3_step(T0, x0, f, DT/2, m, b)
            return (RL_ODE3_step(t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))


def RL_ODE4_step(t0: SupportsFloat,
                 x0: Any,
                 f: Callable[[Any, Any], Any],
                 dt: SupportsFloat,
                 m: Callable[[Any], SupportsFloat],
                 b: None | SupportsFloat = None
                 ) -> tuple[float, Any]:
    """Run the optimal 4th order method (under the Ralston/Lotkin error
    bound judgement scheme) for an ODE
        x' = f(t, x),
    starting at t = t0 and x = x0, with total time step of dt

    If b is not None, the result is constrained by the condition that\n
    each [sub]step gives a change in value of x from the previous step\n
    satisfying the condition
        m(delta x) <= b.

    Butcher tableau:

      0 |

      a | a

      b | c   d

      e | f   f   g

     ---+---------------

        | i   j   k   l

    a = 2/5
        (implemented as 3602879701896397/9007199254740992)

    b = (14 - 3 sqrt(5))/16
        (implemented as 2052458128278593/4503599627370496)

    c = (1428 sqrt(5) - 2889)/1024
        (implemented as 5349873001382255/18014398509481984;\n
         actually closer to 334367062586391/1125899906842624,\n
         but using that value creates an inconsistency of
          c + d - b = 1/18014398509481984
         with the other implementation values)

    d = (3785 - 1620 sqrt(5))/1024
        (implemented as 2859959511732117/18014398509481984)

    e = 1

    f = (2094 sqrt(5) - 3365)/6040
        (implemented as 7857894617148927/36028797018963968)

    g = - (975 + 3046 sqrt(5))/2552
        (implemented as -109922604054190831/36028797018963968;\n
         actually closer to -6870162753386927/2251799813685248,\n
         but using that value creates an inconsistency of
          f + g + h - e = -1/36028797018963968
         with the other implementation values)

    h = (467040 + 203968 sqrt(5))/240845
        (implemented as 8630844153500367/2251799813685248)

    i = (263 + 24 sqrt(5))/1812
        (implemented as 787050342077415/4503599627370496)

    j = (125 - 1000 sqrt(5))/3828
        (implemented as -9934592431370783/18014398509481984;\n
         actually closer to -310456013480337/562949953421312,\n
         but using that value creates an inconsistency of
          i + j + k + l - 1 = -1/18014398509481984
         with the other implementation values)

    k = (3426304 + 1661952 sqrt(5))/5924787
        (implemented as 5429249676224051/4503599627370496)

    l = (30 - 4 sqrt(5))/123
        (implemented as 3083790867646903/18014398509481984)
    """
    T0: float = float(t0)
    DT: float = float(dt)
    k1 = f(T0, x0)
    k2 = f(T0 + ((0xccccccccccccd/(1 << 53)) * DT),
           x0 + (DT * ((0xccccccccccccd/(1 << 53)) * k1)))
    k3 = f(T0 + ((0x74ab325a70441/(1 << 52)) * DT),
           x0 + (DT * ((0x1301ae5fd7417/(1 << 50)) * k1 +
                       (0xa291e36c4cf95/(1 << 54)) * k2)))
    k4 = f(T0 + DT,
           x0 + (DT * ((0x1beab6a9566dff/(1 << 55)) * k1 -
                       (0x1868606a76f9aef/(1 << 55)) * k2 +
                       (0x1ea9b4ffe192cf/(1 << 51)) * k3)))
    dx = DT * ((0x2cbd1700f53e7/(1 << 52)) * k1 -
               (0x234b758cdc321f/(1 << 54)) * k2 +
               (0x1349dfb2592633/(1 << 52)) * k3 +
               (0xaf4b1033a49b7/(1 << 54)) * k4)
    if b is not None:
        assert (float(b) > 0)
        if float(m(dx)) > float(b):
            t1, x1 = RL_ODE4_step(T0, x0, f, DT/2, m, b)
            return (RL_ODE4_step(t1, x1, f, DT/2, m, b))
    return ((T0 + DT, x0 + dx))
