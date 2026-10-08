from .alphabeta import AlphaBetaAgent
from .base import Agent
from .belief_mc import BeliefMCAgent
from .greedy_agent import GreedyAgent
from .heuristic_agents import HeuristicAgent, SummonAttackAgent
from .random_agent import RandomAgent
from .simple_agents import MixedAgent, SetterAgent

AGENTS = {"random": RandomAgent, "greedy": GreedyAgent, "setter": SetterAgent, "mixed": MixedAgent,
          "belief_mc": BeliefMCAgent, "summon_attack": SummonAttackAgent, "heuristic": HeuristicAgent,
          "alphabeta": AlphaBetaAgent}


def make_agent(name: str, seed: int | None = None) -> Agent:
    return AGENTS[name](seed=seed)
