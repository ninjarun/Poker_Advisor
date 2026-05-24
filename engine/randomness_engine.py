from __future__ import annotations

import random
from typing import Optional


def should_bluff(ev: float, pot_size: float, equity: float, bluff_frequency: float = 0.2) -> bool:
    return RandomnessEngine(bluff_frequency=bluff_frequency).should_bluff(ev, pot_size, equity)


def mixed_action(primary_action: str, secondary_action: str, primary_weight: float) -> str:
    return RandomnessEngine().mixed_action(primary_action, secondary_action, primary_weight)


class RandomnessEngine:
    """Controlled mixed-strategy randomization.

    Kept isolated so the bot does not become deterministic and so future exploit
    protection/counterfactual sampling can be upgraded without touching EV logic.
    """

    def __init__(self, bluff_frequency: float = 0.2, rng: Optional[random.Random] = None):
        self.bluff_frequency = max(0.0, min(bluff_frequency, 1.0))
        self._rng = rng or random.SystemRandom()

    def should_bluff(self, ev: float, pot_size: float, equity: float) -> bool:
        if pot_size <= 0:
            return False
        # Semi-bluff only: enough equity to continue, not so much that this is value.
        if ev < 0 and 0.15 <= equity <= 0.45:
            return self._rng.random() < self.bluff_frequency
        return False

    def mixed_action(self, primary_action: str, secondary_action: str, primary_weight: float) -> str:
        weight = max(0.0, min(primary_weight, 1.0))
        return primary_action if self._rng.random() < weight else secondary_action
