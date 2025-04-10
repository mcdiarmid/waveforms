import numpy as np
from numpy.typing import NDArray


DEFAULT_RNG = np.random.Generator(np.random.PCG64(seed=1))


def generate_complex_awgn(
    sigma: float,
    size: int,
    rng: np.random.Generator = None,
) -> NDArray[np.complex128]:
    rng = rng or DEFAULT_RNG
    return (
        rng.normal(
            loc=0,
            scale=sigma * np.sqrt(2) / 2,
            size=(size, 2),
        )
        .view(np.complex128)
        .flatten()
    )
