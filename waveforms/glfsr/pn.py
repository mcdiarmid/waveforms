from __future__ import annotations

from waveforms.glfsr.glfsr import GLFSR

GALOIS_LFSR_POLYS: list[int] = [  # https://docs.amd.com/v/u/en-US/xapp052
    [0],
    [1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 3],
    [6, 5],
    [7, 6],
    [8, 6, 5, 4],
    [9, 5],
    [10, 7],
    [11, 9],
    [12, 6, 4, 1],
    [13, 4, 3, 1],
    [14, 5, 3, 1],
    [15, 14],
    [16, 15, 13, 4],
    [17, 14],
    [18, 11],
    [19, 6, 2, 1],
    [20, 17],
    [21, 19],
    [22, 21],
    [23, 18],
    [24, 23, 22, 17],
    [25, 22],
    [26, 6, 2, 1],
    [27, 5, 2, 1],
    [28, 25],
    [29, 27],
    [30, 6, 4, 1],
    [31, 28],
    [32, 22, 2, 1],
    [33, 20],
    [34, 27, 2, 1],
    [35, 33],
    [36, 25],
    [37, 5, 4, 3, 2, 1],
    [38, 6, 5, 1],
    [39, 35],
    [40, 38, 21, 19],
    [41, 38],
    [42, 41, 20, 19],
    [43, 42, 38, 37],
    [44, 43, 18, 17],
    [45, 44, 42, 41],
    [46, 45, 26, 25],
    [47, 42],
    [48, 47, 21, 20],
    [49, 40],
    [50, 49, 24, 23],
    [51, 50, 36, 35],
    [52, 49],
    [53, 52, 38, 37],
    [54, 53, 18, 17],
    [55, 31],
    [56, 55, 35, 34],
    [57, 50],
    [58, 39],
    [59, 58, 38, 37],
    [60, 59],
    [61, 60, 46, 45],
    [62, 61, 6, 5],
    [63, 62],
    [64, 63, 61, 60],
]


def generate_mask(degree: int) -> int:
    """Generates the polynomial mask given the PN degree.

    Args:
        degree (int): Number of stages in the feedback shift register.

    Returns:
        int: LFSR polynomial represented as an integer.

    Raises:
        KeyError: when degree does not have a definition in LFSR_POLYS.
    """
    if not 1 < degree < len(GALOIS_LFSR_POLYS):
        err = f"PRBS Polynomial Not Defined for {degree}."
        raise KeyError(err)
    return sum([(1 << (tap - 1)) for tap in GALOIS_LFSR_POLYS[degree]])


class PNSequence(GLFSR):
    def __init__(self, degree: int) -> None:
        self.degree = degree
        super().__init__(generate_mask(degree), (1 << degree) - 1)

    def generate_sequence(self) -> list[int]:
        """Generates the full GLFSR sequence.

        Args:
            None

        Returns:
            list[int]: Sequence of bits.
        """
        return [self.next_bit() for _ in range(2**self.degree - 1)]


if __name__ == "__main__":
    import numpy as np

    degree = 15
    glfsr = PNSequence(degree)
    pn_bits = [glfsr.next_bit() for _ in range(2**degree - 1)]
    pn_bytes = np.packbits(pn_bits)
