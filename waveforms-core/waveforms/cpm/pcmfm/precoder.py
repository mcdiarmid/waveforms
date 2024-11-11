import numpy as np
from numpy.typing import NDArray


class PCMFMSymbolMapper:
    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Encodes bits to symbols when instance is called.

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.int8]: Output symbols.
        """
        return 2 * bits.astype(np.int8) - 1
