from __future__ import annotations

import logging
from dataclasses import dataclass


_logger = logging.getLogger(__name__)


@dataclass
class Branch:
    inp: int
    out: int
    start: int
    end: int


@dataclass
class Trellis:
    branches: list[list[Branch]]

    @property
    def input_cardinality(self) -> int:
        """Returns the input cardinality (number of bits per input value).

        Args:
            None

        Returns:
            int: Input Cardinality.
        """
        unique_inputs = len({b.inp for col in self.branches for b in col})
        if 1 << (unique_inputs.bit_length() - 1) != unique_inputs:
            err = "Number of unique inputs should be multiple of 2."
            raise ValueError(err)
        return unique_inputs.bit_length() - 1

    @property
    def output_cardinality(self) -> int:
        """Returns the input cardinality (number of unique output values).

        Args:
            None

        Returns:
            int: Output Cardinality.
        """
        return len({b.out for col in self.branches for b in col})

    @property
    def columns(self) -> int:
        """Returns the number of trellis layers/columns.

        Args:
            None

        Returns:
            int: Number of trellis columns.
        """
        return len(self.branches)

    @property
    def branches_per_column(self) -> int:
        """Returns the number of trellis branches per layer/column.

        Args:
            None

        Returns:
            int: Number of trellis branches per column.
        """
        return len(self.branches[0])

    @property
    def states(self) -> int:
        """Returns the number of trellis states/rows.

        Args:
            None

        Returns:
            int: Number of trellis states.
        """
        return max([b.start for col in self.branches for b in col]) + 1


def filter_branches(state: int, branches: list[Branch], attr: str = "start") -> list[Branch]:
    """Filters a list of branches based on a state and matching attribute.

    Args:
        state (int): Trellis state (zero indexed)
        branches (list[Branch]): Branches to be filtered
        attr (str): Filtering attribute

    Returns:
        list[Branch]: Branches matching the filter condition
    """
    return [*filter(lambda b: getattr(b, attr) == state, branches)]


def forward_branches(state: int, branches: list[Branch]) -> list[Branch]:
    """Filters a list of branches that start at input state.

    Args:
        state (int): Trellis state (zero indexed)
        branches (list[Branch]): Branches to be filtered

    Returns:
        list[Branch]: Branches starting at input state
    """
    return filter_branches(state, branches, attr="start")


def reverse_branches(state: int, branches: list[Branch]) -> list[Branch]:
    """Filters a list of branches that end at input state.

    Args:
        state (int): Trellis state (zero indexed)
        branches (list[Branch]): Branches to be filtered

    Returns:
        list[Branch]: Branches ending at input state
    """
    return filter_branches(state, branches, attr="end")


def forward_map(state: int, branches: list[Branch]) -> dict[int, Branch]:
    """Generates a map viable forward state transitions from specified trellis state.

    Args:
        state (int): Trellis state (zero indexed)
        branches (list[Branch]): Branches to be filtered and mapped

    Returns:
        dict[int, Branch]: State transition map - input: branch
    """
    return {branch.inp: branch for branch in filter_branches(state, branches, "start")}


def reverse_map(state: int, branches: list[Branch]) -> dict[int, Branch]:
    """Generates a map viable reverse state transitions from specified trellis state.

    Args:
        state (int): Trellis state (zero indexed)
        branches (list[Branch]): Branches to be filtered and mapped

    Returns:
        dict[int, Branch]: State transition map - output: branch
    """
    return {branch.out: branch for branch in filter_branches(state, branches, "end")}


class FiniteStateMachine:
    def __init__(self, trellis: Trellis) -> None:
        self.trellis = trellis

        # Generate forward and reverse maps for quick access
        self.branches_per_column = trellis.branches_per_column
        self.columns = trellis.columns
        self.states = trellis.states
        self.forward_branch_mapping = [
            [forward_map(st, column) for st in range(self.states)] for column in trellis.branches
        ]
        self.reverse_branch_mapping = [
            [reverse_map(st, column) for st in range(self.states)] for column in trellis.branches
        ]
        self.forward_transitions = [
            [{b.end: b for b in forward_branches(st, column)} for st in range(self.states)]
            for column in trellis.branches
        ]
        self.reverse_transitions = [
            [{b.start: b for b in reverse_branches(st, column)} for st in range(self.states)]
            for column in trellis.branches
        ]
        self.symbols = sorted({branch.out for column in trellis.branches for branch in column})
        self.symbol_idx_map = {sym: i for i, sym in enumerate(self.symbols)}


SOQPSKTrellis8x1 = Trellis(
    branches=[
        [
            # Column 1 (n-even/I)
            Branch(inp=0, out=0, start=0, end=4),
            Branch(inp=1, out=+2, start=0, end=6),
            Branch(inp=0, out=0, start=1, end=5),
            Branch(inp=1, out=-2, start=1, end=7),
            Branch(inp=0, out=-2, start=2, end=4),
            Branch(inp=1, out=0, start=2, end=6),
            Branch(inp=0, out=+2, start=3, end=5),
            Branch(inp=1, out=0, start=3, end=7),
            # Column 2 (n-odd/Q)
            Branch(inp=0, out=0, start=4, end=0),
            Branch(inp=1, out=-2, start=4, end=1),
            Branch(inp=0, out=+2, start=5, end=0),
            Branch(inp=1, out=0, start=5, end=1),
            Branch(inp=0, out=0, start=6, end=2),
            Branch(inp=1, out=+2, start=6, end=3),
            Branch(inp=0, out=-2, start=7, end=2),
            Branch(inp=1, out=0, start=7, end=3),
        ],
    ],
)


SOQPSKTrellis4x2 = Trellis(
    branches=[
        # Column 1 (n-even/I)
        [
            Branch(inp=0, out=0, start=0, end=0),
            Branch(inp=1, out=+2, start=0, end=2),
            Branch(inp=0, out=0, start=1, end=1),
            Branch(inp=1, out=-2, start=1, end=3),
            Branch(inp=0, out=-2, start=2, end=0),
            Branch(inp=1, out=0, start=2, end=2),
            Branch(inp=0, out=+2, start=3, end=1),
            Branch(inp=1, out=0, start=3, end=3),
        ],
        # Column 2 (n-odd/Q)
        [
            Branch(inp=0, out=0, start=0, end=0),
            Branch(inp=1, out=-2, start=0, end=1),
            Branch(inp=0, out=+2, start=1, end=0),
            Branch(inp=1, out=0, start=1, end=1),
            Branch(inp=0, out=0, start=2, end=2),
            Branch(inp=1, out=+2, start=2, end=3),
            Branch(inp=0, out=-2, start=3, end=2),
            Branch(inp=1, out=0, start=3, end=3),
        ],
    ],
)


SOQPSKTrellis4x2DiffEncoded = Trellis(
    branches=[
        # Column 1 (n-even/I)
        [
            Branch(inp=0, out=0, start=0, end=0),
            Branch(inp=1, out=+2, start=0, end=2),
            Branch(inp=0, out=0, start=1, end=1),
            Branch(inp=1, out=-2, start=1, end=3),
            Branch(inp=1, out=-2, start=2, end=0),
            Branch(inp=0, out=0, start=2, end=2),
            Branch(inp=1, out=+2, start=3, end=1),
            Branch(inp=0, out=0, start=3, end=3),
        ],
        # Column 2 (n-odd/Q)
        [
            Branch(inp=0, out=0, start=0, end=0),
            Branch(inp=1, out=-2, start=0, end=1),
            Branch(inp=1, out=+2, start=1, end=0),
            Branch(inp=0, out=0, start=1, end=1),
            Branch(inp=0, out=0, start=2, end=2),
            Branch(inp=1, out=+2, start=2, end=3),
            Branch(inp=1, out=-2, start=3, end=2),
            Branch(inp=0, out=0, start=3, end=3),
        ],
    ],
)

SimpleTrellis2 = Trellis(
    branches=[
        [
            Branch(inp=0, out=-1, start=0, end=0),
            Branch(inp=1, out=+1, start=0, end=1),
            Branch(inp=0, out=-1, start=1, end=0),
            Branch(inp=1, out=+1, start=1, end=1),
        ],
    ],
)
SimpleTrellis4 = Trellis(
    branches=[
        [
            Branch(inp=0, out=-3, start=0, end=0),
            Branch(inp=1, out=-1, start=0, end=1),
            Branch(inp=2, out=+1, start=0, end=2),
            Branch(inp=3, out=+3, start=0, end=3),

            Branch(inp=0, out=-3, start=1, end=0),
            Branch(inp=1, out=-1, start=1, end=1),
            Branch(inp=2, out=+1, start=1, end=2),
            Branch(inp=3, out=+3, start=1, end=3),

            Branch(inp=0, out=-3, start=2, end=0),
            Branch(inp=1, out=-1, start=2, end=1),
            Branch(inp=2, out=+1, start=2, end=2),
            Branch(inp=3, out=+3, start=2, end=3),

            Branch(inp=0, out=-3, start=3, end=0),
            Branch(inp=1, out=-1, start=3, end=1),
            Branch(inp=2, out=+1, start=3, end=2),
            Branch(inp=3, out=+3, start=3, end=3),
        ],
    ],
)
