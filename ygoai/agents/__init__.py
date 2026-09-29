from .base import Agent
from .belief_mc import BeliefMCAgent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent

AGENTS = {"random": RandomAgent, "greedy": GreedyAgent, "belief_mc": BeliefMCAgent}


def make_agent(name: str, seed: int | None = None) -> Agent:
    return AGENTS[name](seed=seed)
