from .base import Agent
from .belief_mc import BeliefMCAgent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent
from .simple_agents import MixedAgent, SetterAgent

# (the agents of the belief checks' test games, scripts/check_belief.py and scripts/suite.py; the
# playing agents are the Rust engine's, crates/ygoai)
AGENTS = {"random": RandomAgent, "greedy": GreedyAgent, "setter": SetterAgent, "mixed": MixedAgent,
          "belief_mc": BeliefMCAgent}


def make_agent(name: str, seed: int | None = None) -> Agent:
    return AGENTS[name](seed=seed)
