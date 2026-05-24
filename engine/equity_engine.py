from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import eval7

_CARD_REPLACEMENTS = {
    "♥": "h",
    "♠": "s",
    "♦": "d",
    "♣": "c",
    "10": "T",
}


def clean_cards(cards: Sequence[str]) -> List[str]:
    """Convert OCR/UI card strings into eval7 format.

    Examples: ``10♥`` -> ``Th``, ``A♠`` -> ``As``.
    Invalid/placeholder cards are left out so callers can decide how to handle
    incomplete OCR state.
    """
    cleaned: List[str] = []
    for card in cards:
        if not card or "?" in card:
            continue
        normalized = str(card).strip()
        for old, new in _CARD_REPLACEMENTS.items():
            normalized = normalized.replace(old, new)
        normalized = normalized.replace(" ", "")
        if len(normalized) >= 2:
            cleaned.append(normalized[0].upper() + normalized[1].lower())
    return cleaned


# Backward-compatible name used by earlier code.
def clean_format(cards: Sequence[str]) -> List[str]:
    return clean_cards(cards)


def to_eval7_cards(cards: Sequence[str]) -> List[eval7.Card]:
    return [eval7.Card(card) for card in clean_cards(cards)]


def _cache_key(hero_cards: Sequence[str], board_cards: Sequence[str], villain_range: str, iterations: int) -> Tuple[Tuple[str, ...], Tuple[str, ...], str, int]:
    return tuple(clean_cards(hero_cards)), tuple(clean_cards(board_cards)), villain_range, iterations


@lru_cache(maxsize=2_048)
def _calculate_equity_cached(
    hero_cards: Tuple[str, ...],
    board_cards: Tuple[str, ...],
    villain_range_str: str,
    iterations: int,
) -> float:
    hero_hand = [eval7.Card(card) for card in hero_cards]
    board = [eval7.Card(card) for card in board_cards]
    villain_range = eval7.HandRange(villain_range_str)
    return float(eval7.py_hand_vs_range_monte_carlo(hero_hand, villain_range, board, iterations))


def calculate_equity(
    hero_cards: Sequence[str],
    board_cards: Sequence[str],
    villain_range_str: str,
    iterations: int = 10_000,
    use_cache: bool = True,
) -> float:
    """Calculate hero equity versus a villain range using eval7 Monte Carlo.

    ``use_cache`` is safe for repeated same-state UI refreshes and can be
    disabled later for exact stochastic sampling or opponent-model experiments.
    """
    hero_key, board_key, range_key, iteration_key = _cache_key(
        hero_cards, board_cards, villain_range_str, iterations
    )
    if len(hero_key) != 2:
        raise ValueError(f"Expected exactly 2 hero cards, got {len(hero_key)}")
    if len(board_key) > 5:
        raise ValueError(f"Expected at most 5 board cards, got {len(board_key)}")

    if use_cache:
        return _calculate_equity_cached(hero_key, board_key, range_key, iteration_key)

    hero_hand = [eval7.Card(card) for card in hero_key]
    board = [eval7.Card(card) for card in board_key]
    villain_range = eval7.HandRange(range_key)
    return float(eval7.py_hand_vs_range_monte_carlo(hero_hand, villain_range, board, iteration_key))


def calculate_range_vs_range_equity(*_args: object, **_kwargs: object) -> float:
    """Reserved extension point for future solver/range-vs-range support."""
    raise NotImplementedError("Range-vs-range equity is not implemented yet.")
