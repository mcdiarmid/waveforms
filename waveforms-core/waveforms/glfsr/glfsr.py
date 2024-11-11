class GLFSR:
    def __init__(self, mask: int, state: int) -> None:
        self.mask = mask
        self.state = state

    def next_bit(self) -> int:
        """Produces the next bit in the GLFSR sequence.

        Args:
            None

        Returns:
            int: 0 or 1, next bit in the sequence.
        """
        bit = self.state & 1
        self.state >>= 1
        if bit:
            self.state ^= self.mask
        return bit
