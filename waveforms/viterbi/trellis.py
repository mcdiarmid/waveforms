import logging
from typing import List, Dict
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class Branch:
    inp: int
    out: int
    start: int
    end: int


@dataclass
class State:
    branches: List[Branch]


@dataclass
class Trellis:
    branches: List[Branch]


class FiniteStateMachine:
    def __init__(self, trellis: Trellis, start: int = 0):
        self.trellis = trellis
        self.state = start
        # TODO check trellis validity

    def forward_branches(self) -> List[Branch]:
        return [*filter(lambda b: b.start == self.state, self.trellis.branches)]

    def backward_branches(self) -> List[Branch]:
        return [*filter(lambda b: b.end == self.state, self.trellis.branches)]

    def forward_map(self) -> Dict[int, int]:
        return {branch.inp: branch.end for branch in self.forward_branches()}

    def backward_map(self) -> Dict[int, int]:
        return {branch.out: branch.start for branch in self.backward_branches()}

    def next(self, inp: int):
        mapping = self.forward_map()
        if inp not in mapping:
            _logger.warning(f"No branch corresponding to input {inp} from starting state {self.state}")
            return
        self.state = mapping[inp]

    def prev(self, out: int):
        mapping = self.backward_map()
        if out not in mapping:
            _logger.warning(f"No branch corresponding to output {out} from ending state {self.state}")
            return
        self.state = mapping[out]


SOQPSKTrellis = Trellis(
    branches=[
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
    ]
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fsm = FiniteStateMachine(trellis=SOQPSKTrellis)
