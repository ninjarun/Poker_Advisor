from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def get_bet_size(pot_size: float, spr: float, hand_tier: int, street: str, board_texture: str = "dry") -> float:
    return SizingEngine().get_bet_size(pot_size, spr, hand_tier, street, board_texture)


def choose_raise_size(amount_to_call: float, pot_size: float, stack_size: Optional[float] = None) -> float:
    return SizingEngine().choose_raise_size(amount_to_call, pot_size, stack_size)


def choose_preflop_raise_size(
    bb_value: float,
    position: str,
    stack_size: Optional[float] = None,
    amount_to_call: float = 0.0,
    action_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    return SizingEngine().choose_preflop_raise_size(
        bb_value=bb_value,
        position=position,
        stack_size=stack_size,
        amount_to_call=amount_to_call,
        action_history=action_history,
    )


class SizingEngine:
    """SPR and texture-aware bet/raise sizing."""

    def choose_preflop_raise_size(
        self,
        bb_value: float,
        position: str,
        stack_size: Optional[float] = None,
        amount_to_call: float = 0.0,
        action_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> float:
        """Return strict pre-flop sizing, isolated from post-flop pot geometry.

        Open-raises are clamped to 2.5x-3.0x BB. Re-raises/3-bets use about
        3x the original raise size. This intentionally does not inspect SPR or
        current pot size; those belong to post-flop sizing only.
        """
        bb = max(float(bb_value or 0.0), 0.01)
        stack = float(stack_size or 0.0)
        open_raise = self._latest_preflop_raise_size(action_history or (), bb)

        if open_raise > bb or amount_to_call > bb:
            # Facing an open/raise: 3-bet to roughly 3x the original raise.
            base_raise = max(open_raise, float(amount_to_call or 0.0), bb)
            raise_size = base_raise * 3.0
        else:
            # Unopened pot: standard open only. Late positions can use 2.5x;
            # earlier/blind opens use 3.0x. Both stay inside the requested
            # 2.5x-3.0x BB clamp.
            multiplier = 2.5 if position in {"CO", "BTN", "SB"} else 3.0
            multiplier = min(max(multiplier, 2.5), 3.0)
            raise_size = bb * multiplier

        if stack > 0:
            raise_size = min(raise_size, stack)
        return round(max(raise_size, 0.0), 2)

    def get_bet_size(
        self,
        pot_size: float,
        spr: float,
        hand_tier: int,
        street: str,
        board_texture: str = "dry",
    ) -> float:
        if pot_size <= 0:
            return 0.0

        texture = board_texture.lower()
        wet_board = any(token in texture for token in ("connected", "two-tone", "monotone"))

        if spr <= 1.0 and hand_tier in {1, 2}:
            return pot_size  # practical all-in/pot commitment signal
        if hand_tier == 1:
            return pot_size * (0.66 if wet_board else 0.50)
        if hand_tier == 2:
            return pot_size * (0.66 if wet_board else 0.33)
        if hand_tier == 4:
            return pot_size * 0.33
        return pot_size * 0.33

    def choose_raise_size(self, amount_to_call: float, pot_size: float, stack_size: Optional[float] = None) -> float:
        if amount_to_call <= 0:
            raise_size = pot_size * 0.66
        else:
            raise_size = pot_size + (amount_to_call * 3.0)
        if stack_size is not None and stack_size > 0:
            return min(raise_size, stack_size)
        return raise_size

    @staticmethod
    def _latest_preflop_raise_size(action_history: Sequence[Dict[str, Any]], bb_value: float) -> float:
        """Best-effort extraction of the latest pre-flop open/raise size."""
        for action in reversed(action_history):
            action_type = str(action.get("type", "") or "").lower()
            bet = float(action.get("bet", action.get("amount", 0.0)) or 0.0)
            if action_type in {"bet", "raise", "3bet", "3-bet", "all-in", "allin"} and bet > 0:
                return bet
            if bet > bb_value:
                return bet
        return 0.0
