from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

DEFAULT_POSITION_RANGES: Dict[str, str] = {
    "UTG": "77+, ATs+, KQs, AQo+",
    "MP": "66+, A9s+, KTs+, QJs, JTs, AJo+, KQo",
    "CO": "44+, A2s+, K9s+, QTs+, JTs, T9s, 98s, ATo+, KJo+",
    "BTN": "22+, A2s+, K2s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, A2o+, K9o+, QTo+, JTo",
    "SB": "22+, A2s+, K7s+, Q9s+, J9s+, T8s+, 98s, A8o+, KTo+, QJo",
    "BB": "22+, A2s+, K2s+, Q7s+, J8s+, T8s+, 97s+, 86s+, A2o+, K9o+, QTo+, JTo",
    "Unknown": "22+, A2s+, K9s+, QTs+, JTs, ATo+, KJo+",
}


def estimate_villain_range(
    amount_to_call: float,
    pot: float,
    position: str = "Unknown",
    action_history: Optional[List[Dict[str, Any]]] = None,
    num_players: int = 6,
) -> str:
    """Estimate current villain range from position and bet size.

    This preserves the previous bet-size buckets while making position/action
    hooks explicit for future VPIP/PFR, profiling, and range narrowing work.
    """
    bet_ratio = amount_to_call / pot if pot > 0 else 0.0

    if bet_ratio >= 1.0:
        return "77+, AJs+, AQo+, KQs"
    if bet_ratio >= 0.66:
        return "55+, A8s+, ATo+, KJs+, QJs, JTs"
    if bet_ratio > 0.0:
        return "22+, A2+, K9s+, KTo+, Q9s+, QTo+, J9s+, T9s, 98s, 87s"

    return DEFAULT_POSITION_RANGES.get(position, DEFAULT_POSITION_RANGES["Unknown"])


def narrow_range_by_action(
    current_range: str,
    action: Optional[Dict[str, Any]] = None,
    pot_size: float = 0.0,
    street: str = "Pre-Flop",
) -> str:
    """Narrow villain range based on the latest observed action.

    Massive bets are treated as strong/polarized and strip weak broadways,
    small pairs, weak aces, and suited connector noise from the eval7 range.
    """
    if not action:
        return current_range

    bet = float(action.get("bet", 0.0) or 0.0)
    action_type = str(action.get("type", "") or ("bet" if bet > 0 else "check")).lower()
    bet_ratio = bet / pot_size if pot_size > 0 else 0.0

    is_postflop = street != "Pre-Flop"
    if action_type in {"all-in", "allin"} or (is_postflop and bet_ratio > 1.5):
        return _massive_bet_range(street)
    if action_type in {"raise", "3bet"} or bet_ratio >= 1.0:
        return "77+, AJs+, AQo+, KQs"
    if action_type in {"bet", "call"} and bet_ratio >= 0.66:
        return "55+, A8s+, ATo+, KJs+, QJs, JTs"
    return current_range


def _massive_bet_range(street: str) -> str:
    """Very tight eval7-compatible range for overbet/all-in pressure."""
    if street == "Pre-Flop":
        return "JJ+, AKs, AKo"
    return "TT+, AQs+, AKo, KQs"
