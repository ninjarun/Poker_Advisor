from __future__ import annotations

from typing import Optional


def get_bet_size(pot_size: float, spr: float, hand_tier: int, street: str, board_texture: str = "dry") -> float:
    return SizingEngine().get_bet_size(pot_size, spr, hand_tier, street, board_texture)


def choose_raise_size(amount_to_call: float, pot_size: float, stack_size: Optional[float] = None) -> float:
    return SizingEngine().choose_raise_size(amount_to_call, pot_size, stack_size)


class SizingEngine:
    """SPR and texture-aware bet/raise sizing."""

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
