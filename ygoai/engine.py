"""Runs a built-in agent as an external engine: python -m ygoai.engine random|greedy [seed]."""

import sys

from .agents import make_agent
from .protocol import serve


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "random"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    counter = iter(range(1 << 30))

    def factory():
        return make_agent(name, None if seed is None else seed + next(counter))

    serve(factory)


if __name__ == "__main__":
    main()
