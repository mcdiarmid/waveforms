import numpy as np
from numpy.typing import NDArray


DEFAULT_RNG = np.random.Generator(np.random.PCG64(seed=1))


def generate_complex_awgn(
    sigma: float,
    size: int,
    rng: np.random.Generator = None,
) -> NDArray[np.complex128]:
    """Generates complext AWGN.

    Args:
        sigma: Amplitude standard deviation.
        size: Number of samples.
        rng: Random number generator.

    Returns:
        NDArray[np.complex128]: Array of samples containing AWGN.
    """
    rng = rng or DEFAULT_RNG
    return (
        rng.normal(
            loc=0,
            scale=sigma,
            size=(size, 2),
        )
        .view(np.complex128)
        .flatten()
    )
