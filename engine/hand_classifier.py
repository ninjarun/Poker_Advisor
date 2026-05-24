from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import eval7


@dataclass(frozen=True, slots=True)
class DrawInfo:
    has_flush_draw: bool = False
    has_oesd: bool = False
    has_combo_draw: bool = False

    @property
    def has_any_draw(self) -> bool:
        return self.has_flush_draw or self.has_oesd or self.has_combo_draw


@dataclass(frozen=True, slots=True)
class BoardTexture:
    is_paired: bool = False
    is_monotone: bool = False
    is_two_tone: bool = False
    is_connected: bool = False
    high_card_rank: int = 0

    @property
    def label(self) -> str:
        parts: List[str] = []
        if self.is_paired:
            parts.append("paired")
        if self.is_monotone:
            parts.append("monotone")
        elif self.is_two_tone:
            parts.append("two-tone")
        if self.is_connected:
            parts.append("connected")
        return ", ".join(parts) if parts else "dry"


def analyze_board_texture(board: Sequence[eval7.Card]) -> BoardTexture:
    ranks = sorted({card.rank for card in board})
    suits = [card.suit for card in board]
    is_paired = len({card.rank for card in board}) < len(board)
    is_monotone = len(board) >= 3 and any(suits.count(suit) >= 3 for suit in set(suits))
    is_two_tone = len(board) >= 3 and any(suits.count(suit) == 2 for suit in set(suits))
    is_connected = _has_straight_window(ranks, window=3) if len(ranks) >= 3 else False
    high_card_rank = max(ranks) if ranks else 0
    return BoardTexture(is_paired, is_monotone, is_two_tone, is_connected, high_card_rank)


def detect_draws(cards: Sequence[eval7.Card]) -> DrawInfo:
    """Detect common flop/turn draws from combined hero+board cards."""
    suits = [card.suit for card in cards]
    ranks = sorted({card.rank for card in cards})
    # Wheel support: Ace can complete A-2-3-4-5.
    if 14 in ranks:
        ranks = sorted(set(ranks + [1]))

    has_flush_draw = any(suits.count(suit) == 4 for suit in set(suits))
    has_oesd = _has_open_ended_straight_draw(ranks)
    return DrawInfo(
        has_flush_draw=has_flush_draw,
        has_oesd=has_oesd,
        has_combo_draw=has_flush_draw and has_oesd,
    )


def classify_hand(hero_hand: Sequence[eval7.Card], board: Sequence[eval7.Card]) -> Tuple[str, int]:
    """Classify made hand/draw into tier 1-5.

    Tiers:
    1 monster, 2 strong, 3 marginal, 4 draw, 5 trash.
    """
    all_cards = list(hero_hand) + list(board)
    if len(hero_hand) != 2:
        return "Unknown", 5

    # hand_value = eval7.evaluate(all_cards)
    # hand_type = eval7.handtype(hand_value)
    # board_texture = analyze_board_texture(board)

    # board_ranks = sorted([card.rank for card in board], reverse=True)
    # hero_ranks = sorted([card.rank for card in hero_hand], reverse=True)
    # top_board = board_ranks[0] if board_ranks else 0
    # all_ranks = [card.rank for card in all_cards]
    # rank_counts = {rank: all_ranks.count(rank) for rank in set(all_ranks)}
    # pair_ranks = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
    # hero_pair_ranks = [rank for rank in pair_ranks if rank in hero_ranks]

    # hand_tier = 5
    # if hand_type in {"Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"}:
    #     hand_tier = 1
    # elif hand_type == "Two Pair":
    #     # Do not count two-pair that is entirely on the board as Hero's made hand.
    #     if not hero_pair_ranks:
    #         hand_type = "Board Two Pair"
    #         hand_tier = 5
    #     else:
    #         hand_tier = 3 if board_texture.is_paired else 1
    # elif hand_type == "Pair":
    #     # eval7 reports a board-only paired board as "Pair". Only treat it as
    #     # Hero's pair if at least one hero card contributes to the paired rank.
    #     if not hero_pair_ranks:
    #         hand_type = "Board Pair"
    #         hand_tier = 5
    #     else:
    #         pair_rank = max(hero_pair_ranks)
    #         kicker = max(rank for rank in hero_ranks if rank != pair_rank) if any(rank != pair_rank for rank in hero_ranks) else pair_rank
    #         is_overpair = bool(board_ranks) and pair_rank > top_board and hero_ranks.count(pair_rank) == 2
    #         is_top_pair = bool(board_ranks) and pair_rank == top_board
    #         hand_tier = 2 if (is_overpair or (is_top_pair and kicker >= 8)) else 3
    hand_value = eval7.evaluate(all_cards)
    hand_type = eval7.handtype(hand_value)
    board_texture = analyze_board_texture(board)

    # --- 1. Are we just playing the board? (River Check) ---
    is_playing_board = False
    if len(board) == 5:
        if hand_value == eval7.evaluate(board):
            is_playing_board = True

    # --- 2. Extract specific ranks to map what Hero actually holds ---
    board_ranks = sorted([card.rank for card in board], reverse=True)
    hero_ranks = sorted([card.rank for card in hero_hand], reverse=True)
    top_board = board_ranks[0] if board_ranks else 0
    all_ranks = [card.rank for card in all_cards]
    rank_counts = {rank: all_ranks.count(rank) for rank in set(all_ranks)}
    
    pair_ranks = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
    hero_pair_ranks = [rank for rank in pair_ranks if rank in hero_ranks]
    
    trip_ranks = sorted([rank for rank, count in rank_counts.items() if count >= 3], reverse=True)
    hero_trip_ranks = [rank for rank in trip_ranks if rank in hero_ranks]
    
    quad_ranks = sorted([rank for rank, count in rank_counts.items() if count >= 4], reverse=True)
    hero_quad_ranks = [rank for rank in quad_ranks if rank in hero_ranks]

    hand_tier = 5

    # --- 3. Evaluate the Hand Tier ---
    
    # Intercept hands made entirely by the 5 community cards (Straights, Full Houses, etc.)
    if is_playing_board:
        hand_type = f"Board {hand_type}"
        hand_tier = 5

    elif hand_type in {"Straight Flush", "Full House", "Straight"}:
        hand_tier = 1

    elif hand_type == "Quads":
        if not hero_quad_ranks:
            hand_type = "Board Quads"
            hand_tier = 5
        else:
            hand_tier = 1

    elif hand_type == "Flush":
        suits = [c.suit for c in all_cards]
        flush_suit = next((s for s in set(suits) if suits.count(s) >= 5), None)
        
        if flush_suit is not None:
            hero_flush_cards = [c.rank for c in hero_hand if c.suit == flush_suit]
            board_flush_cards = [c.rank for c in board if c.suit == flush_suit]
            
            if not hero_flush_cards:
                hand_type = "Board Flush"
                hand_tier = 5
            elif len(hero_flush_cards) == 1:
                # 1-Card Flush logic: Only nut and 2nd-nut are Tier 1
                best_hero_flush_card = max(hero_flush_cards)
                missing_ranks = [r for r in range(12, -1, -1) if r not in board_flush_cards]
                nut_rank = missing_ranks[0] if missing_ranks else -1
                second_nut_rank = missing_ranks[1] if len(missing_ranks) > 1 else -1
                
                if best_hero_flush_card >= second_nut_rank:
                    hand_tier = 1
                else:
                    hand_type = "Weak Flush"
                    hand_tier = 3
            else:
                # Hero holds 2 cards to the flush
                hand_tier = 1
        else:
            hand_tier = 1

    elif hand_type == "Trips":
        if not hero_trip_ranks:
            hand_type = "Board Trips"
            hand_tier = 5
        else:
            hand_tier = 1

    elif hand_type == "Two Pair":
        if not hero_pair_ranks:
            hand_type = "Board Two Pair"
            hand_tier = 5
        else:
            hand_tier = 3 if board_texture.is_paired else 1

    elif hand_type == "Pair":
        if not hero_pair_ranks:
            hand_type = "Board Pair"
            hand_tier = 5
        else:
            pair_rank = max(hero_pair_ranks)
            kicker = max((rank for rank in hero_ranks if rank != pair_rank), default=pair_rank)
            is_overpair = bool(board_ranks) and pair_rank > top_board and hero_ranks.count(pair_rank) == 2
            is_top_pair = bool(board_ranks) and pair_rank == top_board
            hand_tier = 2 if (is_overpair or (is_top_pair and kicker >= 8)) else 3
            

    if len(board) < 5:
        draws = detect_draws(all_cards)
        if hand_tier >= 3 and draws.has_any_draw:
            hand_tier = 4
            if draws.has_combo_draw:
                hand_type = "Combo Draw"
            elif draws.has_flush_draw:
                hand_type = "Flush Draw"
            elif draws.has_oesd:
                hand_type = "Open-Ended Straight Draw"

    return hand_type, hand_tier


def _has_straight_window(ranks: Sequence[int], window: int) -> bool:
    unique = sorted(set(ranks))
    return any(unique[i + window - 1] - unique[i] <= window for i in range(len(unique) - window + 1))


def _has_open_ended_straight_draw(ranks: Sequence[int]) -> bool:
    unique = sorted(set(ranks))
    if len(unique) < 4:
        return False
    # Four unique ranks spanning exactly 3 gaps means any card on either end completes.
    return any(unique[i + 3] - unique[i] == 3 for i in range(len(unique) - 3))
