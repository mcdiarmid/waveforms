class GLFSR:
    def __init__(self, mask: int, state: int):
        self.mask = mask
        self.state = state

    def next_bit(self):
        bit = self.state & 1
        self.state >>= 1
        if bit:
            self.state ^= self.mask
        return bit
