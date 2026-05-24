from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .equity_engine import calculate_equity, clean_cards, to_eval7_cards
from .ev_engine import calculate_ev, calculate_pot_odds, calculate_spr, minimum_defense_frequency
from .hand_classifier import DrawInfo, analyze_board_texture, classify_hand, detect_draws
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

        # Unknown opponents should use the baseline TAG range until profiling is reliable.
        profile_to_use = "TAG"

        if hasattr(game_state, 'villain_profile'):
            # Basic map handling for raw profile strings
            if "CallingStation" in str(game_state.villain_profile) or "Maniac" in str(game_state.villain_profile):
                profile_to_use = "LOOSE"
            elif "TAG" in str(game_state.villain_profile):
                profile_to_use = "TAG"

        decision, tier, reason = self._preflop_chart_decision(
            hand_code,
            game_state.position,
            game_state.amount_to_call,
            opponent_profile=profile_to_use
        )

        if self._is_late_position_unraised(game_state):
            tier = min(tier, self._late_position_steal_tier(hand_code))

        recommended_size = None
        if self._should_blind_steal(game_state, tier):
            decision = "RAISE"
            reason = "Blind steal: late-position unopened pot with playable stealing hand."
            recommended_size = self._choose_preflop_raise_size(game_state)
        elif decision in {"RAISE", "ALL-IN"}:
            recommended_size = self._choose_preflop_raise_size(game_state)

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
        # Value hands already raise via the chart. This branch exists to force
        # the widened CO/BTN steal range to open instead of passively checking.
        return self._is_late_position_unraised(game_state) and 3 <= hand_tier <= 4

    def _choose_preflop_raise_size(self, game_state: GameState) -> float:
        stack = game_state.stack_size or game_state.effective_stack
        if game_state.amount_to_call > game_state.bb_value:
            size = self.sizing_engine.choose_raise_size(
                game_state.amount_to_call,
                game_state.pot_size,
                stack,
            )
        else:
            multiplier = 2.5 if game_state.position in {"CO", "BTN", "SB"} else 3.0
            size = multiplier * game_state.bb_value
        return round(min(size, stack), 2) if stack > 0 else round(size, 2)

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
        draw_info = detect_draws(list(hero_cards) + list(board_cards)) if len(board_cards) < 5 else DrawInfo()

        if self._should_fit_or_fold_multiway(game_state, hand_tier, draw_info):
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
        decision, reason = self._postflop_decision(
            game_state,
            equity,
            ev,
            pot_odds,
            spr,
            hand_tier,
            hand_type,
            draw_info,
            hero_cards,
            board_cards,
        )

        recommended_size = None
        if self._should_continuation_bet(game_state, board_texture.label):
            decision = "RAISE"
            reason = "Continuation bet: Hero has aggressor advantage on favorable flop texture."
            recommended_size = min(game_state.pot_size * 0.33, game_state.stack_size) if game_state.stack_size > 0 else game_state.pot_size * 0.33
        elif decision == "ALL-IN":
            recommended_size = game_state.stack_size if game_state.stack_size > 0 else game_state.effective_stack
        elif decision == "RAISE":
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
                "draw_info": {
                    "flush_draw": draw_info.has_flush_draw,
                    "oesd": draw_info.has_oesd,
                    "combo_draw": draw_info.has_combo_draw,
                    "backdoor_flush_draw": self._has_backdoor_flush_draw(hero_cards, board_cards, draw_info),
                    "gutshot": self._has_gutshot(hero_cards, board_cards, draw_info),
                    "two_overcards": self._has_two_overcards(hero_cards, board_cards),
                },
            },
        )

    @staticmethod
    def _apply_multiway_equity_penalty(raw_equity: float, num_players: int) -> float:
        if num_players <= 2:
            return raw_equity
        multiplier = max(0.55, 1.0 - (0.12 * (num_players - 2)))
        return min(max(raw_equity * multiplier, 0.0), 1.0)

    @staticmethod
    def _should_fit_or_fold_multiway(game_state: GameState, hand_tier: int, draw_info: DrawInfo) -> bool:
        return (
            game_state.num_players > 2
            and game_state.street != "Pre-Flop"
            and game_state.amount_to_call > 0
            and hand_tier not in {1, 2}
            and not DecisionEngine._is_strong_draw(draw_info)
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
        hand_type: str,
        draw_info: DrawInfo,
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
    ) -> Tuple[str, str]:
        to_call = game_state.amount_to_call
        mdf = minimum_defense_frequency(to_call, game_state.pot_size)

        monster_made_hands = {"Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"}
        nutted_made_hands = {"Full House", "Quads", "Straight Flush"}
        value_raise_hands = monster_made_hands | {"Two Pair"}

        if to_call <= 0:
            if hand_type in nutted_made_hands and spr <= 2.0:
                return "ALL-IN", "Low SPR nutted hand: jam for maximum value."
            if hand_tier == 1:
                return "RAISE", "Value bet monster made hand when checked to."
            if hand_tier == 2 and equity >= 0.50:
                return "RAISE", "Value bet strong made hand when checked to."
            if hand_tier == 4 and self.randomness_engine.should_bluff(ev, game_state.pot_size, equity):
                return "RAISE", "Mixed-frequency semi-bluff with draw equity."
            return "CHECK", "Free card/showdown available; no call required."

        if hand_type in nutted_made_hands:
            if spr <= 2.5 or game_state.stack_size <= to_call * 2.0:
                return "ALL-IN", "Nutted made hand at low SPR: jam instead of slow-playing."
            return "RAISE", "Nutted made hand: raise for value."

        if hand_tier == 1 and hand_type in value_raise_hands:
            if spr <= 1.5:
                return "ALL-IN", "Low SPR monster hand: jam for value/protection."
            if equity >= max(0.40, pot_odds):
                return "RAISE", "Monster made hand: raise for value/protection instead of flat-calling."

        if hand_tier == 2 and spr <= 1.25 and equity >= max(0.45, pot_odds):
            return "ALL-IN", "Low SPR strong hand: commit stack for value."

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

        if self._should_mdf_float(game_state, hand_tier, draw_info, hero_cards, board_cards):
            defend_action = self.randomness_engine.mixed_action("CALL", "FOLD", mdf)
            if defend_action == "CALL":
                if self._should_mdf_check_raise_bluff(game_state, draw_info, hero_cards, board_cards):
                    return "RAISE", "MDF defense: check-raise bluff with blocker/backdoor equity instead of over-folding to TAG pressure."
                return "CALL", "MDF defense: float flop with backdoor equity/overcards/gutshot to avoid fit-or-fold exploitation."

        return "FOLD", "Fold: equity/EV does not justify continuing and hand is below MDF defense candidates."

    @staticmethod
    def _is_strong_draw(draw_info: DrawInfo) -> bool:
        return draw_info.has_combo_draw or draw_info.has_flush_draw or draw_info.has_oesd

    def _should_mdf_float(
        self,
        game_state: GameState,
        hand_tier: int,
        draw_info: DrawInfo,
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
    ) -> bool:
        if game_state.street != "Flop" or game_state.amount_to_call <= 0 or game_state.num_players > 2:
            return False
        if hand_tier in {1, 2, 4}:
            return False
        return (
            self._has_backdoor_flush_draw(hero_cards, board_cards, draw_info)
            or self._has_two_overcards(hero_cards, board_cards)
            or self._has_gutshot(hero_cards, board_cards, draw_info)
        )

    def _should_mdf_check_raise_bluff(
        self,
        game_state: GameState,
        draw_info: DrawInfo,
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
    ) -> bool:
        if game_state.pot_size <= 0 or game_state.stack_size <= game_state.amount_to_call * 3.0:
            return False
        has_good_bluff_properties = self._has_gutshot(hero_cards, board_cards, draw_info) or self._has_backdoor_flush_draw(
            hero_cards, board_cards, draw_info
        )
        return has_good_bluff_properties and self.randomness_engine.mixed_action("RAISE", "CALL", 0.15) == "RAISE"

    @staticmethod
    def _has_backdoor_flush_draw(
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
        draw_info: DrawInfo,
    ) -> bool:
        if len(board_cards) != 3 or draw_info.has_flush_draw:
            return False
        suits = [getattr(card, "suit", None) for card in list(hero_cards) + list(board_cards)]
        return any(suits.count(suit) >= 3 for suit in set(suits) if suit is not None)

    @staticmethod
    def _has_two_overcards(hero_cards: Sequence[object], board_cards: Sequence[object]) -> bool:
        if len(hero_cards) != 2 or not board_cards:
            return False
        board_high = max(getattr(card, "rank", -1) for card in board_cards)
        return all(getattr(card, "rank", -1) > board_high for card in hero_cards)

    @staticmethod
    def _has_gutshot(
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
        draw_info: DrawInfo,
    ) -> bool:
        if len(board_cards) < 3 or draw_info.has_oesd:
            return False
        ranks = {getattr(card, "rank", -99) for card in list(hero_cards) + list(board_cards)}
        ranks.discard(-99)
        if len(ranks) < 4:
            return False

        if max(ranks) <= 12:  # eval7-style 0..12 ranks, ace high is 12.
            windows = [set(range(start, start + 5)) for start in range(0, 9)]
            windows.append({12, 0, 1, 2, 3})
        else:
            windows = [set(range(start, start + 5)) for start in range(2, 11)]
            windows.append({14, 2, 3, 4, 5})

        for window in windows:
            present = ranks & window
            if len(present) == 4:
                missing = next(iter(window - present))
                if missing not in {min(window), max(window)}:
                    return True
        return False

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

    def _preflop_chart_decision(
        self,
        hand_code: str,
        position: str,
        amount_to_call: float,
        opponent_profile: str = "TAG",
    ) -> Tuple[str, int, str]:
        premium = {"AA", "KK", "QQ", "JJ", "AKs"}
        strong = {"TT", "99", "88", "AKo", "AQs", "AQo", "AJs", "KQs"}

        # Base Sets
        suited_aces = {f"A{rank}s" for rank in "KQJT98765432"}
        suited_kings = {f"K{rank}s" for rank in "QJT98765432"}
        suited_connectors_54_plus = {"KQs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s"}
        suited_one_gappers = {"KJs", "QTs", "J9s", "T8s", "97s", "86s", "75s", "64s"}
        broadways = {"AKo", "AQo", "AJo", "ATo", "KQo", "KJo", "KTo", "QJo", "QTo", "JTo"}
        offsuit_connectors = {"KQo", "QJo", "JTo", "T9o", "98o"}
        three_bet_bluffs = {"A5s", "A4s", "A3s", "A2s", "T9s", "98s", "87s", "76s", "65s", "54s"}

        # ---- STRICT TAG RANGES (Original) ----
        if opponent_profile == "TAG":
            playable = {"77", "66", "55", "ATs", "KJs", "QTs", "JTs", "T9s", "98s", "87s"}
            if position in {"UTG", "MP"}:
                playable = {"77", "66", "ATs", "KJs", "QTs", "JTs"}
            elif position in {"CO", "BTN", "SB"}:
                playable |= suited_aces | suited_connectors_54_plus | suited_one_gappers | broadways | {"44", "33", "22"}
                strong |= {"ATo", "KQo", "AJo"}

        # ---- LOOSE / EXPLOITATIVE RANGES (New) ----
        elif opponent_profile in {"WEAK", "CallingStation", "LOOSE"}:
            playable = {"77", "66", "55", "44", "33", "22"} | suited_connectors_54_plus

            if position in {"UTG", "MP"}:
                playable |= suited_aces | broadways

            elif position in {"CO", "BTN", "SB"}:
                playable |= suited_aces | suited_kings | suited_connectors_54_plus | suited_one_gappers | broadways | offsuit_connectors
                strong |= {"ATo", "KQo", "AJo"}

            elif position == "BB" and amount_to_call > 0:
                playable |= suited_aces | suited_kings | broadways | offsuit_connectors

        else:
            playable = {"77", "66", "55", "ATs", "KJs", "QTs", "JTs", "T9s", "98s", "87s"}

        # ---- DECISION LOGIC ----
        if hand_code in premium:
            return "RAISE", 1, "Premium preflop hand: raise or 3-bet for value."

        if amount_to_call > 0 and hand_code in three_bet_bluffs:
            # Drop bluff freq against Calling stations as they don't fold
            bluff_freq = 0.10 if opponent_profile in {"WEAK", "CallingStation", "LOOSE"} else 0.25
            if self.randomness_engine.mixed_action("RAISE", "CALL", bluff_freq) == "RAISE":
                return "RAISE", 4, f"Polarized 3-bet bluff: mixed at {int(bluff_freq * 100)}%."
            if hand_code in playable:
                return "CALL", 4, "Mixed 3-bet bluff not selected; calling with equity."

        if hand_code in strong:
            return "RAISE" if amount_to_call <= 0 else "CALL", 2, "Strong preflop hand in continuing range."

        if hand_code in playable:
            return "RAISE" if amount_to_call <= 0 else "CALL", 3, "Playable preflop hand for position/action."

        if amount_to_call > 0:
            return "FOLD", 5, "Outside preflop continuing range facing a bet."

        if position == "BB":
            return "CHECK", 5, "Outside opening range, checking the free option."

        return "FOLD", 5, "Outside opening range; fold preflop."
