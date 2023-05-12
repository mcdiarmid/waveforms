from typing import List, Tuple, Literal, Dict

import numpy as np
from numpy.typing import NDArray
from numpy import cos, sin, exp


"""From IRIG 106 2022 Chapter 2.3.3.2
n Cn an ωn θn (deg)
Fourth-order Bessel filter (N=2)
1 1.321405 -0.995209 1.257106 126.547182
2 4.049108 -1.370068 0.410250 -78.794227

Fifth-order Bessel filter (N=2)
0 8.594891 -1.502316 0.717910 -152.010950
1 1.261186 -0.957677 1.471124 60.587298
2 5.568066 -1.380877 0.717910 -152.010950

Sixth-order Bessel filter (N=3)
1 1.174080 -0.930657 1.661863 -3.657188
2 7.027081 -1.381858 0.971472 138.041767
3 15.136528 -1.571490 0.320896 -74.465371

Seventh-order Bessel filter (N=3)
0 32.675401 -1.684368 0.589245 -146.065944
1 1.070723 -0.909868 1.836451 -66.775408
2 8.331228 -1.378903 1.191567 70.220403
3 23.598522 -1.612039 0.589245 -146.065944

Eighth-order Bessel filter (N=4)
1 0.959732 -0.892870 1.998326 -129.100317
2 9.414294 -1.373841 1.388357 3.883967
3 33.701870 -1.636939 0.822796 144.350081
4 60.742083 -1.757408 0.272868 -72.170321
"""


BesselParams = List[Tuple[float, float, float, float]]

BESSEL_FOURTH_ORDER_PARAMS = [
    (1.321405, -0.995209, 1.257106, 126.547182),
    (4.049108, -1.370068, 0.410250, -78.794227),
]
BESSEL_FIFTH_ORDER_PARAMS = [
    (8.594891, -1.502316, 0.717910, -152.010950),
    (1.261186, -0.957677, 1.471124, 60.587298),
    (5.568066, -1.380877, 0.717910, -152.010950),
]
BESSEL_SIXTH_ORDER_PARAMS = [
    (1.174080, -0.930657, 1.661863, -3.657188),
    (7.027081, -1.381858, 0.971472, 138.041767),
    (15.136528, -1.571490, 0.320896, -74.465371),
]
BESSEL_SEVENTH_ORDER_PARAMS = [
    (32.675401, -1.684368, 0.589245, -146.065944),
    (1.070723, -0.909868, 1.836451, -66.775408),
    (8.331228, -1.378903, 1.191567, 70.220403),
    (23.598522, -1.612039, 0.589245, -146.065944),
]
BESSEL_EIGHTH_ORDER_PARAMS = [
    (0.959732, -0.892870, 1.998326, -129.100317),
    (9.414294, -1.373841, 1.388357, 3.883967),
    (33.701870, -1.636939, 0.822796, 144.350081),
    (60.742083, -1.757408, 0.272868, -72.170321),
]
BESSEL_PARAMS_LUT: Dict[int, BesselParams] = {
    4: BESSEL_FOURTH_ORDER_PARAMS,
    5: BESSEL_FIFTH_ORDER_PARAMS,
    6: BESSEL_SIXTH_ORDER_PARAMS,
    7: BESSEL_EIGHTH_ORDER_PARAMS,
    8: BESSEL_EIGHTH_ORDER_PARAMS,
}


def _bessel_b(
    x: NDArray[np.float64],
    c: float,
    a: float,
    omega: float,
    theta: float,
) -> NDArray[np.float64]:
    b = 2 * np.pi * 0.7
    # theta *= np.pi / 180  # deg to rad
    return 2 * c * exp(a * b * x) * (
        (omega * sin(theta) - a * cos(theta)) * cos(omega * b * x) +
        (a * sin(theta) + omega * cos(theta)) * sin(omega * b * x)
    ) / (a**2 + omega**2)


def bessel(
    sps: int = 8,
    order: Literal[4, 5, 6, 7, 8] = 6
) -> NDArray[np.float64]:
    bessel_params = BESSEL_PARAMS_LUT.get(order)
    if bessel_params is None:
        raise KeyError(f"Invalid Bessel Filter order.  Supported values are: {list(BESSEL_PARAMS_LUT)}.")

    length = 3
    tau = np.linspace(0, length, num=length*sps)
    if order % 2:
        (c, a, _omega, _theta), *bessel_params = bessel_params
        g = c * (exp(a * np.maximum(tau-1, 0)) - exp(a * tau)) / a
    else:
        g = np.zeros(tau.shape, dtype=np.float64)

    return g + sum([
        _bessel_b(tau, *params) - _bessel_b(np.maximum(tau - 1, 0), *params)
        for params in bessel_params
    ])
