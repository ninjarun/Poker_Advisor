from __future__ import annotations


def calculate_ev(equity: float, pot: float, to_call: float) -> float:
    """Expected value for calling/facing a bet, in table currency."""
    equity = max(0.0, min(float(equity), 1.0))
    return (equity * pot) - ((1.0 - equity) * to_call)


def calculate_pot_odds(to_call: float, pot: float) -> float:
    """Required equity to call profitably."""
    denominator = pot + to_call
    return to_call / denominator if denominator > 0 else 0.0


def calculate_spr(eff_stack: float, pot: float) -> float:
    """Stack-to-pot ratio."""
    return eff_stack / pot if pot > 0 else 0.0


def minimum_defense_frequency(to_call: float, pot: float) -> float:
    """Minimum defense frequency versus a bet."""
    return pot / (pot + to_call) if to_call > 0 and (pot + to_call) > 0 else 1.0


# Backward-compatible alias.
def calculate_mdf(to_call: float, pot: float) -> float:
    return minimum_defense_frequency(to_call, pot)
