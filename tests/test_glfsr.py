import pytest

from waveforms.glfsr.pn import PNSequence


@pytest.mark.parametrize(
    "order",
    range(2, 23),
)
def test_glfsr_pn(order: int) -> None:
    """Tests that all PN sequences are only cyclic at maximal length.

    Args:
        order: LFSR order/memory

    Returns:
        None

    Raises:
        AssertationError: when test fails
    """
    glfsr = PNSequence(order)
    sequence1 = [glfsr.next_bit() for _ in range(2**order - 1)]
    sequence2 = [glfsr.next_bit() for _ in range(2**order - 1)]
    assert sequence1 == sequence2
