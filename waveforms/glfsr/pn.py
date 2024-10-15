from __future__ import annotations

from waveforms.glfsr.glfsr import GLFSR

LFSR_POLYS: dict[int, list[int]] = {
    7: [6],
    9: [5],
    11: [9],
    13: [12, 2, 1],
    15: [14],
    17: [14],
    20: [17],
    23: [18],
    29: [27],
    31: [28],
}
DIGIKEY_LFSR_POLYS: dict[int, list[int]] = {
    2: [1],
    3: [2],
    4: [3],
    5: [3],
    6: [5],
    7: [6],
    8: [6, 5, 4],
    9: [5],
    10: [7],
    11: [9],
    12: [6, 4, 1],
    13: [4, 3, 1],
    14: [5, 3, 1],
    15: [14],
    16: [15, 13, 4],
    17: [14],
    18: [11],
    19: [6, 2, 1],
    20: [17],
    21: [19],
    22: [21],
    23: [18],
    24: [23, 22, 17],
    25: [22],
    26: [6, 2, 1],
    27: [5, 2, 1],
    28: [25],
    29: [27],
    30: [6, 4, 1],
    31: [28],
}


def generate_mask(degree: int) -> int:
    """Generates the polynomial mask given the PN degree.

    Args:
        degree (int): Number of stages in the feedback shift register.

    Returns:
        int: LFSR polynomial represented as an integer.

    Raises:
        KeyError: when degree does not have a definition in LFSR_POLYS.
    """
    if degree not in LFSR_POLYS:
        err = f"PRBS Polynomial Not Defined for {degree}."
        raise KeyError(err)
    return sum([(1 << (tap - 1)) for tap in (degree, *LFSR_POLYS[degree])])


class PNSequence(GLFSR):
    def __init__(self, degree: int) -> None:
        super().__init__(generate_mask(degree), (1 << degree) - 1)


if __name__ == "__main__":
    import numpy as np

    degree = 15
    glfsr = PNSequence(degree)
    pn_bits = [glfsr.next_bit() for _ in range(2**degree - 1)]
    pn_bytes = np.packbits(pn_bits)
