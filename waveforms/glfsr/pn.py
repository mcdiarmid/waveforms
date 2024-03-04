from waveforms.glfsr.glfsr import GLFSR


polynomials_masks = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    (1<<6) + (1<<5),
    0,
    (1<<8) + (1<<4),
    0,
    (1<<10) + (1<<8),
    0,
    (1<<12) + (1<<11) + (1<<1) + (1<<0),
    0,
    (1<<14) + (1<<13),
    0,
    0,
    0,
    0,
    (1<<19) + (1<<2),
    0,
    0,
    (1<<22) + (1<<17),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    (1<<30) + (1<<27),
]


class PNSequence(GLFSR):
    def __init__(self, degree: int):
        super().__init__(
            polynomials_masks[degree],
            (1 << degree) - 1
        )


if __name__ == "__main__":
    import numpy as np
    def print_hex(arr):
        print("".join(f"{x:02x}" for x in arr))

    degree = 15
    glfsr = PNSequence(degree)
    pn_bits = [glfsr.next_bit() for _ in range(2**degree-1)]
    pn_bytes = np.packbits(pn_bits)
