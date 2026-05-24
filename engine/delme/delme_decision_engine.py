from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .equity_engine import calculate_equity, clean_cards, to_eval7_cards
from .ev_engine import calculate_ev, calculate_pot_odds, calculate_spr, minimum_defense_frequency
from .hand_classifier import analyze_board_texture, classify_hand
from .models import AnalysisResult, GameState
from .randomness_engine import RandomnessEngine
from .range_engine import estimate_villain_range, narrow_range_by_action
from .sizing_engine import SizingEngine


TIER_LABELS: Dict[int, str] = {
    1: "Monster",
    2: "Strong",
    3: "Marginal",
    4: "Draw",
    5: "Trash",
}


@dataclass(slots=True)
class DecisionEngineConfig:
    equity_iterations: int = 10_000
    use_equity_cache: bool = True
    bluff_frequency: float = 0.20


class DecisionEngine:
    """Coordinates poker-only engines and returns UI-agnostic decisions."""

    def __init__(self, config: Optional[DecisionEngineConfig] = None):
        self.config = config or DecisionEngineConfig()
        self.sizing_engine = SizingEngine()
        self.randomness_engine = RandomnessEngine(bluff_frequency=self.config.bluff_frequency)

    def analyze(self, game_state: GameState) -> AnalysisResult:
        """Analyze the current hand state and return a structured recommendation."""
        if len(clean_cards(game_state.hero_hand)) != 2:
            return AnalysisResult(
                decision="WAIT",
                reason="Waiting for valid hole cards.",
                pot_odds=calculate_pot_odds(game_state.amount_to_call, game_state.pot_size),
                spr=calculate_spr(game_state.effective_stack, game_state.pot_size),
            )

        if game_state.is_preflop:
            return self._analyze_preflop(game_state)
        return self._analyze_postflop(game_state)

    def _analyze_preflop(self, game_state: GameState) -> AnalysisResult:
        hand_code = self._preflop_hand_code(game_state.hero_hand)
        pot_odds = calculate_pot_odds(game_state.amount_to_call, game_state.pot_size)
        spr = calculate_spr(game_state.effective_stack, game_state.pot_size)
        decision, tier, reason = self._preflop_chart_decision(hand_code, game_state.position, game_state.amount_to_call)
        if self._is_late_position_unraised(game_state):
            tier = min(tier, self._late_position_steal_tier(hand_code))

        recommended_size = None
        if self._should_blind_steal(game_state, tier):
            decision = "RAISE"
            reason = "Blind steal: late-position unopened pot with playable stealing hand."
            recommended_size = min(2.5 * game_state.bb_value, game_state.stack_size) if game_state.stack_size > 0 else 2.5 * game_state.bb_value
        elif decision in {"RAISE", "ALL-IN"}:
            recommended_size = self.sizing_engine.choose_raise_size(
                game_state.amount_to_call,
                game_state.pot_size,
                game_state.stack_size or game_state.effective_stack,
            )

        return AnalysisResult(
            decision=decision,
            reason=reason,
            equity=0.0,
            raw_equity=0.0,
            hand_type=hand_code,
            hand_tier=tier,
            pot_odds=pot_odds,
            ev=0.0,
            spr=spr,
            recommended_size=recommended_size,
            metadata={
                "mdf": minimum_defense_frequency(game_state.amount_to_call, game_state.pot_size),
                "tier_label": TIER_LABELS.get(tier, "Unknown"),
            },
        )

    @staticmethod
    def _is_late_position_unraised(game_state: GameState) -> bool:
        if game_state.position not in {"CO", "BTN"}:
            return False
        # Pre-flop, a standard unopened pot means we only face the 1.0bb Big Blind or 0.5bb Small Blind completion
        if game_state.amount_to_call > game_state.bb_value:
            return False
        # Ensure nobody else has bet or raised before us
        if game_state.pot_size > (1.5 * game_state.bb_value):
            return False

        for action in game_state.action_history:
            action_type = str(action.get("type", "") or "").lower()
            bet = float(action.get("bet", 0.0) or 0.0)
            if action_type in {"bet", "raise", "3bet", "all-in", "allin"} or bet > game_state.bb_value:
                return False
        return True

    @staticmethod
    def _late_position_steal_tier(hand_code: str) -> int:
        steal_candidates = {
            "A6s", "A5s", "A4s", "A3s", "A2s",
            "K9s", "K8s", "K7s", "Q9s", "Q8s", "J9s", "J8s",
            "T8s", "97s", "86s", "76s", "65s", "54s",
            "A9o", "A8o", "KJo", "KTo", "QJo", "QTo", "JTo",
        }
        return 4 if hand_code in steal_candidates else 5

    def _should_blind_steal(self, game_state: GameState, hand_tier: int) -> bool:
        return self._is_late_position_unraised(game_state) and hand_tier <= 4

    def _analyze_postflop(self, game_state: GameState) -> AnalysisResult:
        pot_odds = calculate_pot_odds(game_state.amount_to_call, game_state.pot_size)
        spr = calculate_spr(game_state.effective_stack, game_state.pot_size)

        villain_range = estimate_villain_range(
            game_state.amount_to_call,
            game_state.pot_size,
            position=game_state.position,
            action_history=game_state.action_history,
            num_players=game_state.num_players,
        )
        if game_state.action_history:
            villain_range = narrow_range_by_action(
                villain_range,
                action=game_state.action_history[-1],
                pot_size=game_state.pot_size,
                street=game_state.street,
            )

        hero_cards = to_eval7_cards(game_state.hero_hand)
        board_cards = to_eval7_cards(game_state.board)
        hand_type, hand_tier = classify_hand(hero_cards, board_cards)
        board_texture = analyze_board_texture(board_cards)

        if self._should_fit_or_fold_multiway(game_state, hand_tier):
            return AnalysisResult(
                decision="FOLD",
                reason="Multi-way fit-or-fold: fold to postflop bet without Monster or Strong hand.",
                equity=0.0,
                raw_equity=0.0,
                hand_type=hand_type,
                hand_tier=hand_tier,
                pot_odds=pot_odds,
                ev=0.0,
                spr=spr,
                recommended_size=None,
                villain_range=villain_range,
                metadata={
                    "board_texture": board_texture.label,
                    "mdf": minimum_defense_frequency(game_state.amount_to_call, game_state.pot_size),
                    "tier_label": TIER_LABELS.get(hand_tier, "Unknown"),
                    "fit_or_fold_multiway": True,
                },
            )

        calculated_raw_equity = calculate_equity(
            game_state.hero_hand,
            game_state.board,
            villain_range,
            iterations=self.config.equity_iterations,
            use_cache=self.config.use_equity_cache,
        )
        raw_equity = self._apply_multiway_equity_penalty(calculated_raw_equity, game_state.num_players)

        equity = self._apply_equity_realization(raw_equity, hand_tier, game_state.position, game_state.amount_to_call)
        ev = calculate_ev(equity, game_state.pot_size, game_state.amount_to_call)
        decision, reason = self._postflop_decision(game_state, equity, ev, pot_odds, spr, hand_tier)

        recommended_size = None
        if self._should_continuation_bet(game_state, board_texture.label):
            decision = "RAISE"
            reason = "Continuation bet: Hero has aggressor advantage on favorable flop texture."
            recommended_size = min(game_state.pot_size * 0.33, game_state.stack_size) if game_state.stack_size > 0 else game_state.pot_size * 0.33
        elif decision in {"RAISE", "ALL-IN"}:
            recommended_size = self.sizing_engine.get_bet_size(
                game_state.pot_size,
                spr,
                hand_tier,
                game_state.street,
                board_texture.label,
            )
            if game_state.stack_size > 0:
                recommended_size = min(recommended_size, game_state.stack_size)

        return AnalysisResult(
            decision=decision,
            reason=reason,
            equity=equity,
            raw_equity=raw_equity,
            hand_type=hand_type,
            hand_tier=hand_tier,
            pot_odds=pot_odds,
            ev=ev,
            spr=spr,
            recommended_size=recommended_size,
            villain_range=villain_range,
            metadata={
                "board_texture": board_texture.label,
                "mdf": minimum_defense_frequency(game_state.amount_to_call, game_state.pot_size),
                "tier_label": TIER_LABELS.get(hand_tier, "Unknown"),
                "base_raw_equity": calculated_raw_equity,
                "multiway_penalty": raw_equity / calculated_raw_equity if calculated_raw_equity > 0 else 1.0,
            },
        )

    @staticmethod
    def _apply_multiway_equity_penalty(raw_equity: float, num_players: int) -> float:
        if num_players <= 2:
            return raw_equity
        multiplier = max(0.55, 1.0 - (0.12 * (num_players - 2)))
        return min(max(raw_equity * multiplier, 0.0), 1.0)

    @staticmethod
    def _should_fit_or_fold_multiway(game_state: GameState, hand_tier: int) -> bool:
        return (
            game_state.num_players > 2
            and game_state.street != "Pre-Flop"
            and game_state.amount_to_call > 0
            and hand_tier not in {1, 2}
        )

    @staticmethod
    def _should_continuation_bet(game_state: GameState, board_texture: str) -> bool:
        if game_state.street != "Flop":
            return False
        if not game_state.hero_is_aggressor:
            return False
        if game_state.amount_to_call > 0:
            return False
        texture = board_texture.lower()
        favorable_tokens = ("dry", "paired", "ace-high", "king-high")
        hostile_tokens = ("connected", "two-tone", "monotone", "wet")
        return any(token in texture for token in favorable_tokens) or not any(token in texture for token in hostile_tokens)

    def _postflop_decision(
        self,
        game_state: GameState,
        equity: float,
        ev: float,
        pot_odds: float,
        spr: float,
        hand_tier: int,
    ) -> Tuple[str, str]:
        to_call = game_state.amount_to_call

        if to_call <= 0:
            if hand_tier in {1, 2} and equity >= 0.55:
                return "RAISE", "Value bet with strong made hand when checked to."
            if hand_tier == 4 and self.randomness_engine.should_bluff(ev, game_state.pot_size, equity):
                return "RAISE", "Mixed-frequency semi-bluff with draw equity."
            return "CHECK", "Free card/showdown available; no call required."

        if hand_tier in {1, 2} and equity > 0.60:
            if spr <= 1.0:
                return "ALL-IN", "Low SPR value jam with strong equity."
            return "RAISE", "Value raise: strong hand and equity advantage."

        if hand_tier == 4 and equity > 0.35:
            if ev >= 0:
                return "CALL", "Profitable draw call by pot odds."
            if self.randomness_engine.should_bluff(ev, game_state.pot_size, equity):
                return "RAISE", "Controlled semi-bluff mix to avoid predictability."

        if equity > pot_odds and ev >= 0:
            return "CALL", "Call is profitable versus pot odds."

        if hand_tier == 4 and spr > 5.0 and equity >= max(0.25, pot_odds * 0.85):
            return "CALL", "Speculative draw continue with deep SPR implied odds."

        return "FOLD", "Fold: equity/EV does not justify continuing."

    @staticmethod
    def _apply_equity_realization(raw_equity: float, hand_tier: int, position: str, amount_to_call: float) -> float:
        multiplier = 1.0
        if hand_tier == 5:
            multiplier = 0.0 if amount_to_call > 0 else 0.5
        elif position in {"BTN", "CO"}:
            multiplier = 1.05
        elif position in {"SB", "BB"}:
            multiplier = 0.80
        elif position in {"UTG", "MP"}:
            multiplier = 0.90
        return min(max(raw_equity * multiplier, 0.0), 1.0)

    @staticmethod
    def _preflop_hand_code(hero_hand: list[str]) -> str:
        cards = clean_cards(hero_hand)
        if len(cards) != 2:
            return "Unknown"

        rank_values = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
        r1, s1 = cards[0][0].upper(), cards[0][1].lower()
        r2, s2 = cards[1][0].upper(), cards[1][1].lower()
        if rank_values[r2] > rank_values[r1]:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        return f"{r1}{r2}" if r1 == r2 else f"{r1}{r2}{'s' if s1 == s2 else 'o'}"

    @staticmethod
    def _preflop_chart_decision(hand_code: str, position: str, amount_to_call: float) -> Tuple[str, int, str]:
        premium = {"AA", "KK", "QQ", "JJ", "AKs"}
        strong = {"TT", "99", "88", "AKo", "AQs", "AQo", "AJs", "KQs"}
        playable = {"77", "66", "55", "ATs", "KJs", "QTs", "JTs", "T9s", "98s", "87s"}

        if position in {"UTG", "MP"}:
            playable = {"77", "66", "ATs", "KJs", "QTs", "JTs"}
        elif position in {"CO", "BTN"}:
            playable |= {"44", "33", "22", "A9s", "A8s", "A7s", "KTo", "QJo", "86s", "75s"}
            strong |= {"ATo", "KQo", "J9s"}

        if hand_code in premium:
            return "RAISE", 1, "Premium preflop hand: raise or 3-bet for value."
        if hand_code in strong:
            return "RAISE" if amount_to_call <= 0 else "CALL", 2, "Strong preflop hand in continuing range."
        if hand_code in playable:
            return "RAISE" if amount_to_call <= 0 else "CALL", 3, "Playable preflop hand for position/action."
        if amount_to_call > 0:
            return "FOLD", 5, "Outside preflop continuing range facing a bet."
        return "CHECK", 5, "Outside opening range; check if free option exists."
