from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class GameState:
    """Normalized, UI-agnostic table state consumed by poker engines."""

    hero_hand: List[str] = field(default_factory=list)
    board: List[str] = field(default_factory=list)
    pot_size: float = 0.0
    amount_to_call: float = 0.0
    stack_size: float = 0.0
    position: str = "Unknown"
    street: str = "Pre-Flop"
    num_players: int = 6
    current_stacks: Dict[str, float] = field(default_factory=dict)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    statuses: Dict[str, str] = field(default_factory=dict)
    bb_value: float = 1.0
    hero_is_aggressor: bool = False

    def __post_init__(self) -> None:
        if not self.street or self.street == "Unknown":
            self.street = street_from_board(self.board)

    @property
    def effective_stack(self) -> float:
        active_stacks = [stack for stack in self.current_stacks.values() if stack and stack > 0]
        if active_stacks:
            return min(active_stacks)
        return max(self.stack_size, 0.0)

    @property
    def is_preflop(self) -> bool:
        return len(self.board) == 0


@dataclass(slots=True)
class AnalysisResult:
    """Structured decision payload returned by DecisionEngine.

    Engines return plain values only. UI/log rendering belongs to callers.
    """

    decision: str
    reason: str
    equity: float = 0.0
    raw_equity: float = 0.0
    hand_type: str = "Unknown"
    hand_tier: int = 5
    pot_odds: float = 0.0
    ev: float = 0.0
    spr: float = 0.0
    recommended_size: Optional[float] = None
    villain_range: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ev_dollars(self) -> float:
        """Backward-compatible alias for older callers."""
        return self.ev

    @property
    def logic(self) -> str:
        """Backward-compatible alias for older callers."""
        return self.reason


def street_from_board(board: List[str]) -> str:
    board_len = len(board)
    if board_len == 0:
        return "Pre-Flop"
    if board_len == 3:
        return "Flop"
    if board_len == 4:
        return "Turn"
    if board_len == 5:
        return "River"
    return "Unknown"
