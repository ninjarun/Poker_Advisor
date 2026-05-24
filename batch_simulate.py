#!/usr/bin/env python3
"""Headless 6-max batch simulator for the poker DecisionEngine.

Runs randomized 6-max Texas Hold'em hands against modular villain archetypes,
logs every hand into simulation_results.csv, and prints progress every 10% of
completion.

Usage:
    ./batch_simulate.py
    ./batch_simulate.py --hands 10000 --equity-iterations 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import eval7

from engine.decision_engine import DecisionEngine, DecisionEngineConfig
from engine.models import GameState, street_from_board


POSITIONS_6MAX: Tuple[str, ...] = ("UTG", "MP", "CO", "BTN", "SB", "BB")
STARTING_STACK_BB = 100.0
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0
DEFAULT_HANDS = 10_000
DEFAULT_OUTPUT = "simulation_results.csv"
CSV_COLUMNS = [
    "Hand_ID",
    "Hero_Position",
    "Hero_Hole_Cards",
    "Villain_Hole_Cards",
    "Opponent_Profile",
    "Final_Board",
    "Street_Reached",
    "Winner",
    "Action_History",
    "Hero_Contribution",
    "Hero_Winnings",
    "Hero_Net_Profit",
    "Hero_Total_Bankroll",
]


@dataclass(slots=True)
class Action:
    actor: str
    decision: str
    amount: float = 0.0


@dataclass(slots=True)
class PlayerState:
    name: str
    position: str
    hole_cards: List[str]
    stack: float = STARTING_STACK_BB
    profile: Optional["VillainProfile"] = None
    active: bool = True
    all_in: bool = False
    street_bet: float = 0.0
    total_contributed: float = 0.0

    @property
    def is_hero(self) -> bool:
        return self.name == "Hero"

    @property
    def label(self) -> str:
        if self.is_hero:
            return f"Hero({self.position})"
        profile_name = self.profile.name if self.profile else "Unknown"
        return f"{self.position}({profile_name})"


@dataclass(slots=True)
class HandResult:
    winner: str
    street_reached: str
    action_history: str
    hero_contribution: float
    hero_winnings: float
    hero_net_profit: float


class VillainProfile:
    """Base class for modular opponent archetypes."""

    name = "VillainProfile"

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    def act(
        self,
        amount_to_call: float,
        pot_size: float,
        stack_size: float,
        street: str,
        hole_cards: Sequence[str],
        board: Sequence[str],
        position: str = "Unknown",
        action_history: Optional[Sequence[dict]] = None,
    ) -> Action:
        raise NotImplementedError

    def _bet_size(self, pot_size: float, stack_size: float, min_multiplier: float, max_multiplier: float) -> float:
        if stack_size <= 0:
            return 0.0
        base = max(BIG_BLIND_BB, pot_size)
        size = base * self.rng.uniform(min_multiplier, max_multiplier)
        return round(min(max(BIG_BLIND_BB, size), stack_size), 2)

    def _raise_size(self, amount_to_call: float, pot_size: float, stack_size: float, min_multiplier: float, max_multiplier: float) -> float:
        # Total contribution for this action, including the call.
        raise_to = amount_to_call + self._bet_size(pot_size, stack_size, min_multiplier, max_multiplier)
        return round(min(max(amount_to_call, raise_to), stack_size), 2)

    @staticmethod
    def _hand_code(hole_cards: Sequence[str]) -> str:
        if len(hole_cards) != 2:
            return "Unknown"
        ranks = "23456789TJQKA"
        r1, r2 = hole_cards[0][0].upper(), hole_cards[1][0].upper()
        s1, s2 = hole_cards[0][1].lower(), hole_cards[1][1].lower()
        if ranks.index(r2) > ranks.index(r1):
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        return f"{r1}{r2}" if r1 == r2 else f"{r1}{r2}{'s' if s1 == s2 else 'o'}"

    @classmethod
    def _is_premium_preflop(cls, hole_cards: Sequence[str]) -> bool:
        return cls._hand_code(hole_cards) in {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"}

    @classmethod
    def _is_playable_preflop(cls, hole_cards: Sequence[str], position: str) -> bool:
        code = cls._hand_code(hole_cards)
        base = {"99", "88", "77", "66", "AQo", "AJs", "ATs", "KQs", "KJs", "QJs", "JTs", "T9s", "98s"}
        late = {"55", "44", "33", "22", "A9s", "A8s", "A7s", "A5s", "KQo", "KTs", "QTs", "J9s", "87s", "76s"}
        blinds = {"AJo", "ATo", "KJo", "QJo", "K9s", "Q9s", "86s", "75s"}
        if cls._is_premium_preflop(hole_cards) or code in base:
            return True
        if position in {"CO", "BTN"} and code in late:
            return True
        if position in {"SB", "BB"} and code in late | blinds:
            return True
        return False

    @staticmethod
    def _has_premium_showdown_value(hole_cards: Sequence[str], board: Sequence[str]) -> bool:
        if len(hole_cards) != 2 or len(board) < 3:
            return False
        cards = [eval7.Card(card) for card in list(hole_cards) + list(board)]
        hand_type = eval7.handtype(eval7.evaluate(cards))
        return hand_type in {"Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"}

    def _has_premium_hand(self, street: str, hole_cards: Sequence[str], board: Sequence[str], position: str) -> bool:
        if street == "Pre-Flop":
            return self._is_premium_preflop(hole_cards)
        return self._has_premium_showdown_value(hole_cards, board)


class CallingStation(VillainProfile):
    """Loose-passive: calls too much, almost never folds, rarely raises."""

    name = "CallingStation"

    def act(self, amount_to_call: float, pot_size: float, stack_size: float, street: str, hole_cards: Sequence[str], board: Sequence[str], position: str = "Unknown", action_history: Optional[Sequence[dict]] = None) -> Action:
        if stack_size <= 0:
            return Action("Villain", "ALL-IN", 0.0)
        roll = self.rng.random()
        playable = self._is_playable_preflop(hole_cards, position) if street == "Pre-Flop" else True
        if amount_to_call > 0:
            if not playable and roll < 0.25:
                return Action("Villain", "FOLD", 0.0)
            if roll < 0.92:
                return Action("Villain", "CALL", min(amount_to_call, stack_size))
            return Action("Villain", "RAISE", self._raise_size(amount_to_call, pot_size, stack_size, 0.35, 0.65))
        if roll < 0.82:
            return Action("Villain", "CHECK", 0.0)
        return Action("Villain", "RAISE", self._bet_size(pot_size, stack_size, 0.25, 0.50))


class Maniac(VillainProfile):
    """Loose-aggressive: large bets, frequent raises, high bluff pressure."""

    name = "Maniac"

    def act(self, amount_to_call: float, pot_size: float, stack_size: float, street: str, hole_cards: Sequence[str], board: Sequence[str], position: str = "Unknown", action_history: Optional[Sequence[dict]] = None) -> Action:
        if stack_size <= 0:
            return Action("Villain", "ALL-IN", 0.0)
        roll = self.rng.random()
        if amount_to_call > 0:
            if roll < 0.12:
                return Action("Villain", "FOLD", 0.0)
            if roll < 0.38:
                return Action("Villain", "CALL", min(amount_to_call, stack_size))
            return Action("Villain", "RAISE", self._raise_size(amount_to_call, pot_size, stack_size, 0.85, 1.75))
        if roll < 0.72:
            return Action("Villain", "RAISE", self._bet_size(pot_size, stack_size, 0.80, 1.50))
        return Action("Villain", "CHECK", 0.0)


class TAG(VillainProfile):
    """Tight-aggressive: folds weak hands and raises premium ranges."""

    name = "TAG"

    def act(self, amount_to_call: float, pot_size: float, stack_size: float, street: str, hole_cards: Sequence[str], board: Sequence[str], position: str = "Unknown", action_history: Optional[Sequence[dict]] = None) -> Action:
        if stack_size <= 0:
            return Action("Villain", "ALL-IN", 0.0)
        has_premium = self._has_premium_hand(street, hole_cards, board, position)
        playable = self._is_playable_preflop(hole_cards, position) if street == "Pre-Flop" else self._has_premium_showdown_value(hole_cards, board)
        roll = self.rng.random()
        if amount_to_call > 0:
            if has_premium and roll < 0.35:
                return Action("Villain", "RAISE", self._raise_size(amount_to_call, pot_size, stack_size, 0.65, 1.10))
            if not playable or roll < 0.68:
                return Action("Villain", "FOLD", 0.0)
            return Action("Villain", "CALL", min(amount_to_call, stack_size))
        if has_premium and roll < 0.65:
            return Action("Villain", "RAISE", self._bet_size(pot_size, stack_size, 0.50, 0.85))
        return Action("Villain", "CHECK", 0.0)


VILLAIN_PROFILES: Tuple[Type[VillainProfile], ...] = (CallingStation, Maniac, TAG)


class BatchSimulator:
    """Full 6-max simulator that feeds realistic table state into DecisionEngine."""

    def __init__(
        self,
        hands: int = DEFAULT_HANDS,
        equity_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.hands = hands
        self.rng = random.Random(seed)
        self.engine = DecisionEngine(
            DecisionEngineConfig(
                equity_iterations=equity_iterations,
                use_equity_cache=False,
                bluff_frequency=0.20,
            )
        )
        self.hero_total_bankroll = STARTING_STACK_BB

    def run(self, output_path: Path) -> None:
        rows = []
        progress_every = max(1, self.hands // 10)

        for hand_id in range(1, self.hands + 1):
            players, full_board = self._deal_6max_hand()
            hero = self._hero(players)
            result = self._play_hand(players, full_board)
            self.hero_total_bankroll = round(self.hero_total_bankroll + result.hero_net_profit, 2)

            rows.append(
                {
                    "Hand_ID": hand_id,
                    "Hero_Position": hero.position,
                    "Hero_Hole_Cards": " ".join(hero.hole_cards),
                    "Villain_Hole_Cards": self._villain_hole_summary(players),
                    "Opponent_Profile": self._villain_profile_summary(players),
                    "Final_Board": " ".join(full_board),
                    "Street_Reached": result.street_reached,
                    "Winner": result.winner,
                    "Action_History": result.action_history,
                    "Hero_Contribution": f"{result.hero_contribution:.2f}",
                    "Hero_Winnings": f"{result.hero_winnings:.2f}",
                    "Hero_Net_Profit": f"{result.hero_net_profit:.2f}",
                    "Hero_Total_Bankroll": f"{self.hero_total_bankroll:.2f}",
                }
            )

            if hand_id % progress_every == 0 or hand_id == self.hands:
                percent = int(hand_id / self.hands * 100)
                print(
                    f"Progress {percent:>3}% | Hand {hand_id:,}/{self.hands:,} | "
                    f"Hero bankroll: {self.hero_total_bankroll:.2f}bb"
                )

        self._write_csv(output_path, rows)
        print(f"Saved {len(rows):,} hands to {output_path}")

    def _deal_6max_hand(self) -> Tuple[Dict[str, PlayerState], List[str]]:
        deck = eval7.Deck()
        deck.shuffle()
        hero_position = self.rng.choice(POSITIONS_6MAX)
        players: Dict[str, PlayerState] = {}

        for position in POSITIONS_6MAX:
            hole_cards = self._cards_to_strings(deck.deal(2))
            if position == hero_position:
                players[position] = PlayerState(name="Hero", position=position, hole_cards=hole_cards)
            else:
                profile = self._select_villain()
                players[position] = PlayerState(name=f"Villain_{position}", position=position, hole_cards=hole_cards, profile=profile)

        full_board = self._cards_to_strings(deck.deal(5))
        return players, full_board

    def _select_villain(self) -> VillainProfile:
        return self.rng.choice(VILLAIN_PROFILES)(rng=self.rng)

    @staticmethod
    def _cards_to_strings(cards: Iterable[eval7.Card]) -> List[str]:
        return [str(card) for card in cards]

    def _play_hand(self, players: Dict[str, PlayerState], full_board: List[str]) -> HandResult:
        pot = 0.0
        action_history: List[dict] = []
        street_reached = "Pre-Flop"
        hero_winnings = 0.0
        hero_is_aggressor = False

        pot += self._post_blind(players["SB"], SMALL_BLIND_BB, action_history)
        pot += self._post_blind(players["BB"], BIG_BLIND_BB, action_history)

        for street, board in self._streets(full_board):
            street_reached = street
            if street != "Pre-Flop":
                self._reset_street_bets(players)

            start_position = "UTG" if street == "Pre-Flop" else self._first_active_left_of_button(players)
            pot, terminal_winner, hero_aggressive_this_round = self._betting_round(
                players=players,
                street=street,
                board=board,
                pot=pot,
                start_position=start_position,
                action_history=action_history,
                hero_is_aggressor=hero_is_aggressor,
            )
            hero_is_aggressor = hero_is_aggressor or hero_aggressive_this_round

            if terminal_winner:
                hero_winnings = self._award_uncontested_pot(players, pot, terminal_winner)
                return self._result(terminal_winner.label, street_reached, players, action_history, hero_winnings)

            active = self._active_players(players)
            if len(active) <= 1:
                winner = active[0] if active else self._hero(players)
                hero_winnings = self._award_uncontested_pot(players, pot, winner)
                return self._result(winner.label, street_reached, players, action_history, hero_winnings)

            # If every remaining player is all-in, run out to showdown.
            if all(player.all_in for player in active):
                street_reached = "River"
                break

        winners = self._showdown(players, full_board)
        hero_winnings = self._award_showdown_pot(pot, winners)
        winner_label = "Split:" + ",".join(winner.label for winner in winners) if len(winners) > 1 else winners[0].label
        return self._result(winner_label, street_reached, players, action_history, hero_winnings)

    def _post_blind(self, player: PlayerState, amount: float, action_history: List[dict]) -> float:
        blind = round(min(amount, player.stack), 2)
        player.stack = round(player.stack - blind, 2)
        player.street_bet = round(player.street_bet + blind, 2)
        player.total_contributed = round(player.total_contributed + blind, 2)
        player.all_in = player.stack <= 0
        action_history.append({"seat": player.position, "label": player.label, "type": "blind", "bet": blind, "street": "Pre-Flop"})
        return blind

    def _streets(self, full_board: Sequence[str]) -> Iterable[Tuple[str, List[str]]]:
        yield "Pre-Flop", []
        yield "Flop", list(full_board[:3])
        yield "Turn", list(full_board[:4])
        yield "River", list(full_board[:5])

    def _betting_round(
        self,
        players: Dict[str, PlayerState],
        street: str,
        board: List[str],
        pot: float,
        start_position: str,
        action_history: List[dict],
        hero_is_aggressor: bool,
    ) -> Tuple[float, Optional[PlayerState], bool]:
        """Run a proper 6-max betting loop until calls match the highest bet."""
        max_bet = max(player.street_bet for player in players.values())
        acted_since_raise = {position for position, player in players.items() if not player.active or player.all_in}
        hero_aggressive_this_round = False
        order = self._ordered_positions_from(start_position)
        cursor = 0
        safety_counter = 0

        while safety_counter < 300:
            safety_counter += 1
            active_players = self._active_players(players)
            active_not_all_in = [p for p in active_players if not p.all_in]
            if len(active_players) <= 1:
                return round(pot, 2), active_players[0], hero_aggressive_this_round
            if not active_not_all_in:
                return round(pot, 2), None, hero_aggressive_this_round
            if all(
                p.position in acted_since_raise and round(p.street_bet, 2) >= round(max_bet, 2)
                for p in active_not_all_in
            ):
                return round(pot, 2), None, hero_aggressive_this_round

            position = order[cursor % len(order)]
            cursor += 1
            player = players[position]
            if not player.active or player.all_in:
                continue
            if position in acted_since_raise and round(player.street_bet, 2) >= round(max_bet, 2):
                continue

            amount_to_call = round(max(0.0, max_bet - player.street_bet), 2)
            action = self._player_action(
                player=player,
                players=players,
                street=street,
                board=board,
                pot=pot,
                amount_to_call=amount_to_call,
                action_history=action_history,
                hero_is_aggressor=hero_is_aggressor,
            )
            contribution, aggressive = self._apply_action(player, action, amount_to_call, pot, action_history, street)
            pot = round(pot + contribution, 2)

            if aggressive:
                max_bet = max(max_bet, player.street_bet)
                acted_since_raise = {position for position, p in players.items() if not p.active or p.all_in}
                if player.is_hero:
                    hero_aggressive_this_round = True
            acted_since_raise.add(position)

        raise RuntimeError("Betting round exceeded safety limit; action loop did not converge")

    def _player_action(
        self,
        player: PlayerState,
        players: Dict[str, PlayerState],
        street: str,
        board: List[str],
        pot: float,
        amount_to_call: float,
        action_history: List[dict],
        hero_is_aggressor: bool,
    ) -> Action:
        if player.is_hero:
            return self._hero_action(player, players, street, board, pot, amount_to_call, action_history, hero_is_aggressor)
        assert player.profile is not None
        action = player.profile.act(
            amount_to_call=round(min(amount_to_call, player.stack), 2),
            pot_size=round(pot, 2),
            stack_size=round(player.stack, 2),
            street=street,
            hole_cards=player.hole_cards,
            board=board,
            position=player.position,
            action_history=action_history,
        )
        action.actor = player.label
        return action

    def _hero_action(
        self,
        hero: PlayerState,
        players: Dict[str, PlayerState],
        street: str,
        board: List[str],
        pot: float,
        amount_to_call: float,
        action_history: List[dict],
        hero_is_aggressor: bool,
    ) -> Action:
        active = self._active_players(players)
        current_stacks = {player.label: round(player.stack, 2) for player in active}
        statuses = {player.label: "Active" if player.active else "Folded" for player in players.values()}
        game_state = GameState(
            hero_hand=hero.hole_cards,
            board=board,
            pot_size=round(pot, 2),
            amount_to_call=round(min(amount_to_call, hero.stack), 2),
            stack_size=round(hero.stack, 2),
            position=hero.position,
            street=street_from_board(board),
            num_players=len(active),
            current_stacks=current_stacks,
            action_history=list(action_history),
            statuses=statuses,
            bb_value=BIG_BLIND_BB,
            hero_is_aggressor=hero_is_aggressor,
        )
        result = self.engine.analyze(game_state)
        return Action(hero.label, result.decision.upper(), result.recommended_size or 0.0)

    def _apply_action(
        self,
        player: PlayerState,
        action: Action,
        amount_to_call: float,
        pot: float,
        action_history: List[dict],
        street: str,
    ) -> Tuple[float, bool]:
        decision = action.decision.upper()
        amount_to_call = round(min(amount_to_call, player.stack), 2)
        aggressive = False

        if decision == "FOLD":
            player.active = False
            action_history.append({"seat": player.position, "label": player.label, "type": "fold", "bet": 0.0, "street": street})
            return 0.0, False

        if decision in {"WAIT", "CHECK"} and amount_to_call <= 0:
            action_history.append({"seat": player.position, "label": player.label, "type": "check", "bet": 0.0, "street": street})
            return 0.0, False

        if decision == "CALL" or (decision in {"WAIT", "CHECK"} and amount_to_call > 0):
            contribution = amount_to_call
            action_type = "call"
        elif decision == "ALL-IN":
            contribution = player.stack
            action_type = "all-in"
            aggressive = contribution > amount_to_call
        else:
            # RAISE/bet amount is interpreted as this action's total contribution,
            # including any call. It must cover to_call and at least a small bet.
            fallback = max(BIG_BLIND_BB, pot * 0.50)
            contribution = max(amount_to_call, action.amount or min(player.stack, fallback))
            contribution = min(contribution, player.stack)
            action_type = "raise" if amount_to_call > 0 else "bet"
            aggressive = contribution > amount_to_call

        contribution = round(min(contribution, player.stack), 2)
        player.stack = round(player.stack - contribution, 2)
        player.street_bet = round(player.street_bet + contribution, 2)
        player.total_contributed = round(player.total_contributed + contribution, 2)
        player.all_in = player.stack <= 0
        action_history.append({"seat": player.position, "label": player.label, "type": action_type, "bet": contribution, "street": street})
        return contribution, aggressive

    def _showdown(self, players: Dict[str, PlayerState], board: List[str]) -> List[PlayerState]:
        remaining = self._active_players(players)
        scores = {
            player.position: eval7.evaluate([eval7.Card(card) for card in player.hole_cards + board])
            for player in remaining
        }
        best_score = max(scores.values())
        return [player for player in remaining if scores[player.position] == best_score]

    def _award_uncontested_pot(self, players: Dict[str, PlayerState], pot: float, winner: PlayerState) -> float:
        share = round(pot, 2)
        winner.stack = round(winner.stack + share, 2)
        return share if winner.is_hero else 0.0

    def _award_showdown_pot(self, pot: float, winners: Sequence[PlayerState]) -> float:
        if not winners:
            return 0.0
        share = round(pot / len(winners), 2)
        remainder = round(pot - (share * len(winners)), 2)
        hero_winnings = 0.0
        for index, winner in enumerate(winners):
            payout = share + (remainder if index == 0 else 0.0)
            winner.stack = round(winner.stack + payout, 2)
            if winner.is_hero:
                hero_winnings += payout
        return round(hero_winnings, 2)

    @staticmethod
    def _reset_street_bets(players: Dict[str, PlayerState]) -> None:
        for player in players.values():
            player.street_bet = 0.0

    @staticmethod
    def _active_players(players: Dict[str, PlayerState]) -> List[PlayerState]:
        return [players[position] for position in POSITIONS_6MAX if players[position].active]

    @staticmethod
    def _hero(players: Dict[str, PlayerState]) -> PlayerState:
        for player in players.values():
            if player.is_hero:
                return player
        raise RuntimeError("Hero was not seated")

    @staticmethod
    def _ordered_positions_from(start_position: str) -> List[str]:
        start_index = POSITIONS_6MAX.index(start_position)
        return list(POSITIONS_6MAX[start_index:] + POSITIONS_6MAX[:start_index])

    def _first_active_left_of_button(self, players: Dict[str, PlayerState]) -> str:
        # Post-flop starts with the first active seat clockwise left of BTN.
        order = self._ordered_positions_from("SB")
        for position in order:
            if players[position].active and not players[position].all_in:
                return position
        for position in order:
            if players[position].active:
                return position
        return "SB"

    @staticmethod
    def _villain_hole_summary(players: Dict[str, PlayerState]) -> str:
        return " | ".join(
            f"{player.position}:{' '.join(player.hole_cards)}"
            for player in players.values()
            if not player.is_hero
        )

    @staticmethod
    def _villain_profile_summary(players: Dict[str, PlayerState]) -> str:
        return " | ".join(
            f"{player.position}:{player.profile.name if player.profile else 'Unknown'}"
            for player in players.values()
            if not player.is_hero
        )

    @staticmethod
    def _result(
        winner: str,
        street_reached: str,
        players: Dict[str, PlayerState],
        action_history: List[dict],
        hero_winnings: float,
    ) -> HandResult:
        hero = BatchSimulator._hero(players)
        hero_contribution = round(hero.total_contributed, 2)
        hero_winnings = round(hero_winnings, 2)
        history_str = " | ".join(
            f"{a['label']}:{a['type']}:{float(a.get('bet', 0.0)):.2f}"
            for a in action_history
        )
        return HandResult(
            winner=winner,
            street_reached=street_reached,
            action_history=history_str,
            hero_contribution=hero_contribution,
            hero_winnings=hero_winnings,
            hero_net_profit=round(hero_winnings - hero_contribution, 2),
        )

    @staticmethod
    def _write_csv(output_path: Path, rows: List[dict]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a headless 6-max batch simulation against DecisionEngine.")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS, help="Number of hands to simulate. Default: 10,000.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="CSV output path. Default: simulation_results.csv.")
    parser.add_argument("--equity-iterations", type=int, default=100, help="Postflop eval7 Monte Carlo iterations per engine decision. Lower values avoid long batch timeouts.")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducible simulations.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hands <= 0:
        raise SystemExit("--hands must be positive")
    if args.equity_iterations <= 0:
        raise SystemExit("--equity-iterations must be positive")

    simulator = BatchSimulator(hands=args.hands, equity_iterations=args.equity_iterations, seed=args.seed)
    simulator.run(args.output)


if __name__ == "__main__":
    main()
