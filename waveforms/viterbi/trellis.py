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
class Trellis:
    branches: List[Branch]


def filter_branches(
    state: int,
    branches: List[Branch],
    attr: str = "start"
) -> List[Branch]:
    return [*filter(lambda b: getattr(b, attr) == state, branches)]


def forward_branches(state: int, branches: List[Branch]) -> List[Branch]:
    return filter_branches(state, branches, attr="start")


def reverse_branches(state: int, branches: List[Branch]) -> List[Branch]:
    return filter_branches(state, branches, attr="end")


def forward_map(state: int, branches: List[Branch]) -> Dict[int, Branch]:
    return {branch.inp: branch for branch in filter_branches(state, branches, "start")}


def reverse_map(state: int, branches: List[Branch]) -> Dict[int, Branch]:
    return {branch.out: branch for branch in filter_branches(state, branches, "end")}


class FiniteStateMachine:
    def __init__(self, trellis: Trellis, start: int = 0):
        self.trellis = trellis
        self.state = start
        # TODO check trellis validity

        # Generate forward and reverse maps for quick access
        self.states = max(trellis.branches, key=lambda b: b.start).start + 1
        self.forward_branch_mapping = [
            forward_map(s, trellis.branches) for s in range(self.states)
        ]
        self.reverse_branch_mapping = [
            reverse_map(s, trellis.branches) for s in range(self.states)
        ]
        self.forward_transitions = [
            {b.end: b for b in forward_branches(st, trellis.branches)}
            for st in range(self.states)
        ]
        self.reverse_transitions = [
            {b.start: b for b in reverse_branches(st, trellis.branches)}
            for st in range(self.states)
        ]
        self.symbols = sorted(set([branch.out for branch in trellis.branches]))

    def next(self, inp: int):
        if inp not in self.forward_branch_mapping[self.state]:
            _logger.warning(f"No branch corresponding to input {inp} from starting state {self.state}")
            return
        self.state = self.forward_branch_mapping[self.state][inp].end

    def prev(self, out: int):
        if out not in self.reverse_branch_mapping[self.state]:
            _logger.warning(f"No branch corresponding to output {out} from ending state {self.state}")
            return
        self.state = self.reverse_branch_mapping[self.state][out].start


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
