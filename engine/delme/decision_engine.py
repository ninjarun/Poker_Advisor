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
        decision, tier, reason = self._preflop_chart_decision(hand_code, game_state.position, game_state.amount_to_call)
        if self._is_late_position_unraised(game_state):
            tier = min(tier, self._late_position_steal_tier(hand_code))

        recommended_size = None
        if self._should_blind_steal(game_state, tier):
            decision = "RAISE"
            reason = "Blind steal: late-position unopened pot with playable stealing hand."
            recommended_size = self.sizing_engine.choose_preflop_raise_size(
                bb_value=game_state.bb_value,
                position=game_state.position,
                stack_size=game_state.stack_size or game_state.effective_stack,
                amount_to_call=game_state.amount_to_call,
                action_history=game_state.action_history,
            )
        elif decision in {"RAISE", "ALL-IN"}:
            # Pre-flop sizing is deliberately separated from post-flop
            # SPR/pot-geometry sizing. Opens stay 2.5x-3.0x BB; 3-bets are
            # sized around 3x the original raise.
            recommended_size = self.sizing_engine.choose_preflop_raise_size(
                bb_value=game_state.bb_value,
                position=game_state.position,
                stack_size=game_state.stack_size or game_state.effective_stack,
                amount_to_call=game_state.amount_to_call,
                action_history=game_state.action_history,
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
        # Keep premium/value opens on their normal value-raise path; only widen
        # late-position stealing for weaker playable steal candidates.
        return self._is_late_position_unraised(game_state) and 3 <= hand_tier <= 4

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
            board_texture.label,
            draw_info,
            hero_cards,
            board_cards,
        )

        recommended_size = None
        semi_bluff_size = self._semi_bluff_bet_size(game_state, equity, draw_info)
        if decision == "RAISE" and semi_bluff_size is not None:
            recommended_size = semi_bluff_size
        elif self._should_continuation_bet(game_state, board_texture.label):
            decision = "RAISE"
            reason = "Continuation bet: Hero has aggressor advantage on favorable flop texture."
            recommended_size = min(game_state.pot_size * 0.33, game_state.stack_size) if game_state.stack_size > 0 else game_state.pot_size * 0.33
        elif decision == "RAISE" and self._is_fragile_lead(hand_type, hand_tier, board_texture.label, hero_cards, board_cards):
            recommended_size = self._protection_bet_size(game_state, board_texture.label)
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
                "draw_info": {
                    "flush_draw": draw_info.has_flush_draw,
                    "oesd": draw_info.has_oesd,
                    "combo_draw": draw_info.has_combo_draw,
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
        # Do not auto-muck strong draws multi-way. OESDs/flush draws have enough
        # equity to continue and can become semi-bluff candidates later.
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
        board_texture: str,
        draw_info: DrawInfo,
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
    ) -> Tuple[str, str]:
        to_call = game_state.amount_to_call

        if to_call <= 0:
            if self._is_fragile_lead(hand_type, hand_tier, board_texture, hero_cards, board_cards):
                return "RAISE", "Protection/value bet: fragile lead on wet turn/river; deny equity instead of taking free showdown."
            if hand_tier in {1, 2} and equity >= 0.55:
                return "RAISE", "Value bet with strong made hand when checked to."
            if self._should_semi_bluff(game_state, equity, board_texture, draw_info):
                return "RAISE", "Semi-bluff: strong draw plus enough fold equity versus break-even bluff threshold."
            return "CHECK", "Free card/showdown available; marginal hand cannot stand a raise."

        if hand_tier in {1, 2} and equity > 0.60:
            if spr <= 1.0:
                return "ALL-IN", "Low SPR value jam with strong equity."
            return "RAISE", "Value raise: strong hand and equity advantage."

        if self._should_semi_bluff(game_state, equity, board_texture, draw_info):
            return "RAISE", "Semi-bluff raise: strong draw with fold equity; aggression beats fit-or-fold calling."

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
    def _is_strong_draw(draw_info: DrawInfo) -> bool:
        # 9-out flush draws and 8-out open-ended straight draws are the minimum
        # semi-bluff candidates; combo draws are always strong draw candidates.
        return draw_info.has_combo_draw or draw_info.has_flush_draw or draw_info.has_oesd

    def _should_semi_bluff(
        self,
        game_state: GameState,
        equity: float,
        board_texture: str,
        draw_info: DrawInfo,
    ) -> bool:
        """Use fold-equity thresholds to turn strong draws into aggression.

        Break-even fold equity for a pure bluff is risk / (risk + reward):
        half-pot needs about 33%, pot-sized needs 50%. Draw equity reduces the
        amount of immediate folds required, so strong draws can raise when the
        estimated fold equity plus realized draw equity clears the threshold.
        """
        if not self._is_strong_draw(draw_info) or game_state.pot_size <= 0:
            return False

        bluff_size = self._semi_bluff_bet_size(game_state, equity, draw_info)
        if bluff_size is None or bluff_size <= 0:
            return False

        break_even_fold_equity = bluff_size / (game_state.pot_size + bluff_size)
        estimated_fold_equity = self._estimate_fold_equity(game_state, board_texture)
        equity_credit = equity * (0.65 if draw_info.has_combo_draw else 0.50)
        return estimated_fold_equity + equity_credit >= break_even_fold_equity

    @staticmethod
    def _estimate_fold_equity(game_state: GameState, board_texture: str) -> float:
        """Conservative heuristic until opponent-specific fold stats exist."""
        fold_equity = 0.18
        texture = board_texture.lower()
        if game_state.amount_to_call <= 0:
            fold_equity += 0.17  # checked-to stab/raise has more immediate FE
        if game_state.hero_is_aggressor:
            fold_equity += 0.12
        if game_state.position in {"BTN", "CO"}:
            fold_equity += 0.05
        if game_state.num_players > 2:
            fold_equity -= 0.10
        if any(token in texture for token in ("monotone", "connected", "two-tone", "wet")):
            fold_equity -= 0.05
        return min(max(fold_equity, 0.05), 0.60)

    @staticmethod
    def _semi_bluff_bet_size(game_state: GameState, equity: float, draw_info: DrawInfo) -> Optional[float]:
        if not DecisionEngine._is_strong_draw(draw_info) or game_state.pot_size <= 0:
            return None
        # Prefer half-pot pressure for standard strong draws (33% BE). Combo
        # draws can size up closer to full pot (50% BE) because they retain more
        # equity when called.
        if draw_info.has_combo_draw and equity >= 0.40:
            size = game_state.pot_size
        else:
            size = game_state.pot_size * 0.50
        if game_state.amount_to_call > 0:
            size = max(size, game_state.amount_to_call * 3.0)
        if game_state.stack_size > 0:
            size = min(size, game_state.stack_size)
        return round(max(size, 0.0), 2)

    @staticmethod
    def _is_fragile_lead(
        hand_type: str,
        hand_tier: int,
        board_texture: str,
        hero_cards: Sequence[object],
        board_cards: Sequence[object],
    ) -> bool:
        """Detect made hands that want protection on wet turn/river boards."""
        if len(board_cards) < 4:
            return False
        texture = board_texture.lower()
        wet_board = any(token in texture for token in ("connected", "two-tone", "monotone", "wet"))
        if not wet_board:
            return False

        if hand_type == "Two Pair":
            # Bottom/medium two pair is ahead often but vulnerable to many river
            # cards on wet textures; bet for value/protection.
            return True

        if hand_type == "Pair" and hand_tier in {2, 3}:
            return DecisionEngine._has_top_pair_weak_kicker(hero_cards, board_cards)

        return False

    @staticmethod
    def _has_top_pair_weak_kicker(hero_cards: Sequence[object], board_cards: Sequence[object]) -> bool:
        if not hero_cards or not board_cards:
            return False
        board_ranks = sorted((getattr(card, "rank", 0) for card in board_cards), reverse=True)
        hero_ranks = sorted((getattr(card, "rank", 0) for card in hero_cards), reverse=True)
        if not board_ranks or not hero_ranks:
            return False
        top_board_rank = board_ranks[0]
        if top_board_rank not in hero_ranks:
            return False
        kickers = [rank for rank in hero_ranks if rank != top_board_rank]
        kicker = max(kickers) if kickers else min(hero_ranks)
        # eval7 uses compact rank integers; <=8 still represents a weak/middling
        # kicker in this codebase's existing rank comparisons.
        return kicker <= 8

    @staticmethod
    def _protection_bet_size(game_state: GameState, board_texture: str) -> float:
        texture = board_texture.lower()
        multiplier = 0.66 if any(token in texture for token in ("monotone", "connected", "two-tone", "wet")) else 0.50
        size = game_state.pot_size * multiplier
        if game_state.stack_size > 0:
            size = min(size, game_state.stack_size)
        return round(max(size, 0.0), 2)

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
        if position == "BB":
            return "CHECK", 5, "Outside opening range, but Big Blind may check the free option."
        return "FOLD", 5, "Outside opening range; fold preflop because only the Big Blind can check a free option."
