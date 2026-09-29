"""Agent that picks uniformly random valid responses."""

from __future__ import annotations

import random

from ..actions import random_response
from ..carddb import default_db
from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def respond(self, prompt: dict, state: dict | None) -> dict:
        return random_response(prompt, self.rng, default_db())
