import time
import json
import re
from datetime import datetime
from pynput import keyboard
import os
import threading 

# ==========================================
# PHASE 1: EXACT MODULE IMPORTS & SETUP
# ==========================================
from hand_cards import get_my_hand
from opponents import run_once
from dealer import get_dealer_seat
from board_cards import get_live_board
from folded import check_active_status
from action import analyze_game, is_my_turn
from engine.decision_engine import DecisionEngine
from engine.models import GameState as EngineGameState

# Gemini SDK Imports
from google import genai
from google.genai import types

# OpenAI Import (NEW)
from openai import OpenAI

# Rich Library Imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# --- CONFIGURATION Gemini ---
MODEL_ID = "gemini-2.0-flash" 
# --- OPENAI FALLBACK CONFIGURATION (NEW) ---
OPENAI_API_KEY = ""
OPENAI_MODEL_ID = "gpt-4o-mini" # Fast and cost-effective for JSON tasks


class PokerOrchestrator:
    def __init__(self):
        self.hand_active = False
        self.auto_running = False 
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) 
        self.engine = DecisionEngine()
        self.ai_advice = None
        self.math_advice = None
        self.last_engine_result = None
        self.amount_to_call = 0.0
        self.ai_source = "Waiting..."
        self.ai_response_time = 0.0  
        
        self.game_state = {
            "hand_id": 0,
            "dealer_pos": None,
            "positions": {},
            "statuses": {},
            "hero_cards": [],
            "initial_stacks": {},
            "current_stacks": {},
            "board": [],
            "total_pot": 0,
            "action_history": []
        }
        
        self.bb_value = self._prompt_for_bb()
        
        print("--- Stealth Poker Orchestrator Ready ---")
        print("Listening for 'a' (Toggle Auto) | 'p' (Manual Hand) | 'o' (Manual Turn) | 'Esc' to quit")

    def _log_math_to_file(self):
        """Strips Rich formatting tags and logs to a file in a background thread."""
        import threading # Add this import
        import re
        from datetime import datetime
        
        if not self.math_advice:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_advice = re.sub(r'\[/?.*?\]', '', self.math_advice)
        
        log_data = (
            f"\n[{timestamp}] ====== LOCAL MATH ENGINE ======\n"
            f"Hand ID: {self.game_state.get('hand_id')} | Street: {self.get_street_name()}\n"
            f"Cards: {self.game_state.get('hero_cards')} | Board: {self.game_state.get('board')}\n"
            f"Pot: ${self.game_state.get('total_pot', 0)} | To Call: ${self.amount_to_call}\n"
            f"--- Result ---\n"
            f"{clean_advice}\n"
        )
        
        # Fire-and-forget background thread
        def write_it():
            with open("math_interactions.txt", "a", encoding="utf-8") as log_file:
                log_file.write(log_data)
                
        threading.Thread(target=write_it).start()

    
    def calculate_math_logic(self):
        """Run the modular poker decision engine and render UI-facing advice.

        OCR/live orchestration still owns raw state collection. Poker math now
        lives in engine/* and returns a structured AnalysisResult.
        """
        try:
            engine_state = self._build_engine_game_state()
            result = self.engine.analyze(engine_state)
            self.last_engine_result = result

            if result.decision == "WAIT":
                self.math_advice = result.reason
                return

            self.math_advice = self._format_math_result(result)
            self._log_math_to_file()
        except Exception as e:
            self.last_engine_result = None
            self.math_advice = f"Decision Engine Error: {e}"
            self._log_math_to_file()

    def _build_engine_game_state(self):
        """Translate legacy orchestrator dict state into the engine dataclass."""
        current_stacks = self.game_state.get("current_stacks", {}) or {}
        statuses = self.game_state.get("statuses", {}) or {}
        active_players = [seat for seat, status in statuses.items() if status == "Active"]
        hero_stack = float(current_stacks.get("Hero", 0.0) or 0.0)

        return EngineGameState(
            hero_hand=list(self.game_state.get("hero_cards", []) or []),
            board=list(self.game_state.get("board", []) or []),
            pot_size=float(self.game_state.get("total_pot", 0.0) or 0.0),
            amount_to_call=float(self.amount_to_call or 0.0),
            stack_size=hero_stack,
            position=self.game_state.get("positions", {}).get("Hero", "Unknown"),
            street=self.get_street_name(),
            num_players=len(active_players) if active_players else 6,
            current_stacks={k: float(v or 0.0) for k, v in current_stacks.items()},
            action_history=list(self.game_state.get("action_history", []) or []),
            statuses=dict(statuses),
            bb_value=float(self.bb_value or 1.0),
            hero_is_aggressor=self._hero_was_last_aggressor(),
        )

    def _hero_was_last_aggressor(self):
        for action in reversed(self.game_state.get("action_history", []) or []):
            seat = action.get("seat")
            action_type = str(action.get("type", "") or "").lower()
            bet = float(action.get("bet", 0.0) or 0.0)
            if seat == "Hero" and (action_type in {"raise", "bet", "all-in", "allin"} or bet > 0):
                return True
            if seat != "Hero" and (action_type in {"raise", "bet", "all-in", "allin"} or bet > 0):
                return False
        return False

    def _should_request_ai_advice(self):
        # """Gate expensive LLM calls to complex spots only."""
        # result = self.last_engine_result
        # if not result or result.decision == "WAIT":
        #     return False
        # amount_to_call_bb = float(self.amount_to_call or 0.0) / float(self.bb_value or 1.0)
        # return amount_to_call_bb > 10.0 or result.hand_tier <= 3
        return False

    def _format_math_result(self, result):
        """UI-only formatting for Rich dashboard/log output."""
        decision_color = {
            "FOLD": "red",
            "CHECK": "green",
            "CALL": "yellow",
            "RAISE": "green",
            "ALL-IN": "bold red",
            "WAIT": "dim",
        }.get(result.decision, "white")
        decision = f"[{decision_color}]{result.decision}[/]"
        if result.recommended_size:
            decision += f" ${result.recommended_size:.2f}"

        tier_label = result.metadata.get("tier_label", "Unknown") if result.metadata else "Unknown"

        if not self.game_state.get("board"):
            return (
                f"Pos: [bold magenta]{self.game_state.get('positions', {}).get('Hero', 'Unknown')}[/] "
                f"| Hand: [bold]{result.hand_type}[/]\n"
                f"Chart Decision: {decision}\n"
                f"Reason: {result.reason}\n"
                f"[dim]Break-Even Bluffing: 33% (1/2 Pot) | 50% (Pot)[/dim]"
            )

        ev_color = "green" if result.ev >= 0 else "red"
        sizing_line = f"\nSizing: [bold cyan]${result.recommended_size:.2f}[/]" if result.recommended_size else ""
        texture = result.metadata.get("board_texture", "unknown") if result.metadata else "unknown"

        return (
            f"Hand: [bold magenta]{result.hand_type}[/] (Tier {result.hand_tier}: {tier_label})\n"
            f"Eq: [bold cyan]{result.equity:.1%}[/] (Raw: {result.raw_equity:.1%}) | Odds: {result.pot_odds:.1%}\n"
            f"EV: [bold {ev_color}]${result.ev:.2f}[/] | SPR: {result.spr:.1f}\n"
            f"Board: [dim]{texture}[/] | Range: [dim]{result.villain_range}[/]\n"
            f"Math Decision: {decision}\n"
            f"Reason: {result.reason}"
            f"{sizing_line}"
        )


    def _prompt_for_bb(self):
        console.clear()
        console.print(Panel("[bold cyan]STEALTH POKER ASSISTANT INITIALIZATION[/]", expand=False))
        while True:
            try:
                val = input("\n💰 Enter the Big Blind amount for this table (e.g., 0.02, 2, 5): ")
                if float(val) > 0:
                    return float(val)
                print("[!] Big Blind must be greater than 0.")
            except ValueError:
                print("[!] Invalid input. Please enter a number.")
    # ==========================================
    # PHASE 6: AUTOMATION STATE MACHINE (NEW)
    # ==========================================
    def toggle_automation(self):
        """Toggles the background automation thread on and off."""
        if self.auto_running:
            self.auto_running = False
            console.print("\n[bold red]⏸️ AUTOMATION STOPPED.[/]")
        else:
            self.auto_running = True
            console.print("\n[bold green]▶️ AUTOMATION STARTED.[/] Waiting for next hand...")
            threading.Thread(target=self._automation_loop, daemon=True).start()

    def _automation_loop(self):
        """Optimized to catch immediate turns on BTN/CO/MP."""
        missed_hand_counter = 0 
        last_card_check_time = 0 
        
        while self.auto_running:
            # --- STATE 1: WAITING FOR NEW HAND ---
            if not self.hand_active:
                current_cards = self.sanitize_cards(get_my_hand())
                if current_cards and '??' not in current_cards:
                    self.start_new_hand() 
                    self.hand_active = True 
                    missed_hand_counter = 0
                    last_card_check_time = time.time()
                    
                    # NEW: Immediate check. If we are BTN/CO, the buttons are likely already there.
                    if is_my_turn():
                        self.process_turn()
                else:
                    time.sleep(0.5)
                continue 

            # --- STATE 2: IN HAND / WAITING FOR TURN ---
            if is_my_turn():
                self.process_turn() 
                while is_my_turn() and self.auto_running:
                    time.sleep(0.2)
                last_card_check_time = time.time()
                continue 

            # ... (Rest of your 3-second slow poll logic remains the same)
            time.sleep(0.1)

            # 2. SLOW POLL: Did the hand end or change? 
            current_time = time.time()
            if current_time - last_card_check_time > 3.0: # Check every 3 seconds
                current_cards = self.sanitize_cards(get_my_hand())
                
                # CASE A: Cards are unreadable/missing (Could be a glitch, or you folded)
                if not current_cards or '??' in current_cards:
                    missed_hand_counter += 1
                    if missed_hand_counter >= 2: # 6 seconds of blank screen = hand definitely over
                        self.hand_active = False
                        missed_hand_counter = 0
                        
                # CASE B: Cards are perfectly readable, but DIFFERENT! (100% a New Hand)
                elif current_cards != self.game_state.get("hero_cards", []):
                    self.hand_active = False # Instantly kill old state!
                    missed_hand_counter = 0
                    # The loop immediately circles back to State 1 and catches the new hand instantly
                    
                # CASE C: Cards are exactly the same (Still in the same hand)
                else:
                    missed_hand_counter = 0
                    
                last_card_check_time = time.time()
            
            # Ultra-fast polling delay
            time.sleep(0.1)

    # ==========================================
    # HELPER FUNCTIONS (Parsing & Formatting)
    # ==========================================
    def parse_opponents_data(self, raw_data):
        formatted_stacks = {}
        pot = 0.0
        if isinstance(raw_data, dict):
            try:
                pot = float(raw_data.get("total_pot", 0.0))
            except (ValueError, TypeError): pass
            if "opponents" in raw_data and isinstance(raw_data["opponents"], list):
                for player in raw_data["opponents"]:
                    if isinstance(player, dict) and "seat" in player and "stack" in player:
                        val = player["stack"]
                        val = val[0] if isinstance(val, list) else val
                        try: formatted_stacks[player["seat"]] = float(val)
                        except (ValueError, TypeError): pass
            else:
                for k, v in raw_data.items():
                    if k != "total_pot":
                        val = v[0] if isinstance(v, list) else v
                        try: formatted_stacks[k] = float(val)
                        except (ValueError, TypeError): pass
        return formatted_stacks, pot

    def sanitize_cards(self, raw_cards):
        if not raw_cards: return []
        if isinstance(raw_cards, str):
            return [c.strip() for c in re.split(r'[\s,]+', raw_cards) if c.strip()]
        if isinstance(raw_cards, list):
            flat = []
            for item in raw_cards:
                if isinstance(item, list) and len(item) > 0: flat.append(str(item[0]).strip())
                elif isinstance(item, str) and item.strip(): flat.append(item.strip())
            return flat
        return []

    def get_street_name(self):
        count = len(self.game_state["board"])
        if count == 0: return "Pre-Flop"
        if count == 3: return "Flop"
        if count == 4: return "Turn"
        if count >= 5: return "River"
        return f"Unknown ({count})"

    def assign_positions(self):
        seats = ["Hero", "Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5"]
        positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        self.game_state["positions"] = {}
        dealer = self.game_state.get("dealer_pos")
        if dealer not in seats: return
        try:
            d_idx = seats.index(dealer)
            rotated = positions[-d_idx:] + positions[:-d_idx]
            for i, seat_name in enumerate(seats):
                self.game_state["positions"][seat_name] = rotated[i]
        except (ValueError, TypeError): pass

    def sync_statuses(self):
        try:
            for seat in self.game_state["current_stacks"].keys():
                self.game_state["statuses"][seat] = "Folded"
            self.game_state["statuses"]["Hero"] = "Active"
            active_list = check_active_status()
            if not isinstance(active_list, list): return
            for active_player in active_list:
                matched_seat = str(active_player).replace("_", " ")
                if matched_seat in self.game_state["statuses"]:
                    self.game_state["statuses"][matched_seat] = "Active"
        except Exception: pass

    def display_rich_dashboard(self):
        console.clear()
        console.print(Panel(f"[bold gold1]GTO POKER ASSISTANT[/] | No Limit Hold'em (BB: ${self.bb_value})", expand=False))
        
        info_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        info_table.add_column("Street")
        info_table.add_column("Pot")
        info_table.add_column("To Call") # Live via action.py
        info_table.add_column("Hole Cards", style="bold yellow")
        info_table.add_column("Board", style="bold white")
        
        board_str = " ".join(self.game_state["board"]) if self.game_state["board"] else "[None]"
        hole_cards_str = " ".join(self.game_state["hero_cards"]) if self.game_state["hero_cards"] else "[None]"
        
        pot_bb = round(self.game_state["total_pot"] / self.bb_value, 2)
        pot_display = f"${self.game_state['total_pot']} [dim]({pot_bb} BB)[/dim]"
        
        # Format Call Amount
        call_bb = round(self.amount_to_call / self.bb_value, 2)
        call_display = f"[bold red]${self.amount_to_call}[/bold red] [dim]({call_bb} BB)[/dim]" if self.amount_to_call > 0 else "[green]CHECK[/green]"

        info_table.add_row(self.get_street_name(), pot_display, call_display, hole_cards_str, board_str)

        player_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        player_table.add_column("Seat")
        player_table.add_column("Pos", justify="center")
        player_table.add_column("Stack", justify="right")
        player_table.add_column("Status")

        for seat, stack_val in self.game_state["current_stacks"].items():
            pos = self.game_state["positions"].get(seat, "-")
            pos_str = "[bold yellow]BTN[/]" if pos == "BTN" else f"[cyan]{pos}[/]"
            status = self.game_state["statuses"].get(seat, "Active")
            status_str = f"[green]{status}[/]" if status == "Active" else f"[dim]{status}[/]"
            stack_bb = round(stack_val / self.bb_value, 1)
            stack_display = f"${stack_val} [dim]({stack_bb} BB)[/dim]"
            player_table.add_row(str(seat), pos_str, stack_display, status_str)

        if self.game_state["action_history"]:
            history_lines = [f"{i}. Seat {a['seat']} bet ${a['bet']} [dim]({round(a['bet']/self.bb_value, 2)} BB)[/dim]" for i, a in enumerate(self.game_state["action_history"], start=1)]
            history_str = "\n".join(history_lines)
        else:
            history_str = "[dim]No actions recorded this street.[/dim]"
            
        history_panel = Panel(history_str, title="[bold white]Action History Tracker[/]", border_style="blue")
        
        console.print(info_table)
        console.print(player_table)
        console.print(history_panel)

        if self.ai_advice:
            if "error" in self.ai_advice:
                ai_str = f"[bold red]API Error:[/] {self.ai_advice['error']}"
            else:
                move = self.ai_advice.get("decision", "UNKNOWN")
                sizing = self.ai_advice.get("raise_percentage", "")
                logic = self.ai_advice.get("thought_process", "No logic provided.")
                move_display = f"[bold green]{move}[/]" if move in ["CALL", "CHECK"] else f"[bold red]{move}[/]"
                if sizing and move in ["RAISE", "ALL-IN"]:
                    move_display += f" [yellow]({sizing})[/]"
                ai_str = f"🎯 [bold white]RECOMMENDED MOVE:[/] {move_display}\n\n🧠 [bold cyan]GTO LOGIC:[/] {logic}"
                
            # --- NEW DYNAMIC TITLE WITH TIMER ---
            panel_title = f"[bold gold1]{self.ai_source} Expert Analysis[/]"
            if self.ai_response_time > 0:
                panel_title += f" [dim white]({self.ai_response_time:.2f}s)[/]"
                
            console.print(Panel(ai_str, title=panel_title, border_style="gold1"))
            
        console.print(Panel(f"[bold cyan]Local Math Advice:[/]\n{self.math_advice}", box=box.DOUBLE))
        console.print("\n[dim italic]Waiting for next state change ('p' or 'o')...[/]")

    # ==========================================
    # PHASE 5: THE AI REQUEST (GEMINI PRIMARY)
    # ==========================================
    def request_ai_advice(self):
        # 2D: Unknown Cards Safety Check
        if "??" in self.game_state["hero_cards"]:
            self.ai_advice = {"error": "OCR failed to read hole cards. AI request aborted."}
            return

        def translate_for_ai(cards):
            mapping = {'♥': 'h', '♠': 's', '♦': 'd', '♣': 'c'}
            translated = []
            for card in cards:
                new_card = card
                for symbol, letter in mapping.items():
                    new_card = new_card.replace(symbol, letter)
                translated.append(new_card)
            return translated

        action_history_bb = [{"seat": a["seat"], "bet_bb": round(a["bet"] / self.bb_value, 2)} for a in self.game_state["action_history"]]
        
        # Calculate Effective Stack
        hero_stack = self.game_state["current_stacks"].get("Hero", 0.0)
        active_opp_stacks = [self.game_state["current_stacks"][s] for s, status in self.game_state["statuses"].items() if status == "Active" and s != "Hero"]
        
        if active_opp_stacks:
            eff_stack_raw = min(hero_stack, max(active_opp_stacks))
        else:
            eff_stack_raw = hero_stack
            
        ai_payload = {
            "table_info": {
                "street": self.get_street_name(),
                "total_pot_bb": round(self.game_state["total_pot"] / self.bb_value, 2),
                "amount_to_call_bb": round(self.amount_to_call / self.bb_value, 2), 
                "effective_stack_bb": round(eff_stack_raw / self.bb_value, 2), 
                "big_blind_value": self.bb_value
            },
            "cards": {
                "hero_hole_cards": translate_for_ai(self.game_state["hero_cards"]),
                "community_board": translate_for_ai(self.game_state["board"])
            },
            "hand_action_history": action_history_bb,
            "active_players": []
        }
        
        for seat, status in self.game_state["statuses"].items():
            if status == "Active":
                stack_bb = round(self.game_state["current_stacks"].get(seat, 0.0) / self.bb_value, 2)
                ai_payload["active_players"].append({
                    "seat": seat,
                    "position": self.game_state["positions"].get(seat, "Unknown"),
                    "stack_bb": stack_bb
                })

        prompt = (
            "You are an elite GTO Poker Bot. Analyze this state. "
            f"Game State JSON (All in BBs):\n{json.dumps(ai_payload, indent=2)}\n\n"
            "State your thought process and decision."
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("ai_interactions.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{timestamp}] ====== AI REQUEST (GEMINI PRIMARY) ======\n{json.dumps(ai_payload, indent=2)}\n")

        start_time = time.time()  # <--- NEW: Start the clock

        # --- 1. TRY GEMINI FIRST ---
        try:
            res = self.client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "thought_process": {"type": "string"},
                            "decision": {"type": "string", "enum": ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]},
                            "raise_percentage": {"type": "string"}
                        },
                        "required": ["thought_process", "decision"]
                    },
                    system_instruction="Analyze exactly the cards provided (e.g. Ah 8s). Identify if suited or offsuit first. If RAISE, provide sizing."
                )
            )
            self.ai_advice = json.loads(res.text)
            self.ai_source = "Gemini (2.0 Flash)" 
            self.ai_response_time = time.time() - start_time  # <--- NEW: Stop clock on success
            
            with open("ai_interactions.txt", "a", encoding="utf-8") as f: 
                f.write(f"\n[{timestamp}] ====== GEMINI RESPONSE ======\n{json.dumps(self.ai_advice, indent=2)}\n")

        except Exception as e:
            # If Gemini fails (rate limits, etc.), trigger OpenAI fallback
            error_msg = str(e).lower()
            with open("ai_interactions.txt", "a", encoding="utf-8") as f: 
                f.write(f"\n[{timestamp}] ====== GEMINI FAILED: {error_msg} -> TRIGGERING OPENAI FALLBACK ======\n")
            
            # Pass start_time to fallback
            self._fallback_to_openai(prompt, timestamp, start_time)

    def _fallback_to_openai(self, prompt, timestamp, start_time):
        """Triggers when Gemini fails or is rate-limited."""
        try:
            system_instruction = (
                "You are an elite GTO Poker Bot. Analyze exactly the cards provided (e.g. Ah 8s). "
                "Identify if suited or offsuit first. If RAISE, provide sizing. "
                "You MUST respond with valid JSON strictly adhering to this schema: "
                '{"thought_process": "string", "decision": "FOLD" | "CHECK" | "CALL" | "RAISE" | "ALL-IN", "raise_percentage": "string"}. '
                "CRITICAL: Keep your thought_process EXTREMELY brief (under 15 words)."
            )

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
                # <--- REMOVED max_tokens completely to prevent JSON severing
            )
            
            raw_content = response.choices[0].message.content
            
            # Temporary debug line to see exactly what OpenAI returns in the terminal
            # print(f"DEBUG - OpenAI Raw Output: {raw_content}") 

            self.ai_advice = json.loads(raw_content)
            self.ai_source = "OpenAI (GPT-4o-mini)" 
            self.ai_response_time = time.time() - start_time  
            
            with open("ai_interactions.txt", "a", encoding="utf-8") as f: 
                f.write(f"\n[{timestamp}] ====== OPENAI FALLBACK RESPONSE ======\n{json.dumps(self.ai_advice, indent=2)}\n")
                
        except Exception as fallback_error:
            self.ai_response_time = time.time() - start_time  
            self.ai_advice = {"error": f"Gemini failed, and OpenAI fallback also failed: {str(fallback_error)}"}

    def start_new_hand(self):
        self.hand_active = True
        self.game_state["hand_id"] += 1
        self.game_state["board"] = []
        
        # --- MISSING STATE RESETS ADDED BACK ---
        self.game_state["action_history"] = []
        self.game_state["statuses"] = {}
        self.ai_advice = None
        self.math_advice = None 
        self.last_engine_result = None
        self.amount_to_call = 0.0
        self.ai_response_time = 0.0  
        # ---------------------------------------

        self.game_state["hero_cards"] = self.sanitize_cards(get_my_hand())
        
        # Start the heavy OCR (stacks/dealer) in background so we don't block the Turn check
        threading.Thread(target=self._initialize_hand_data, daemon=True).start()

    def _initialize_hand_data(self):
        self.game_state["dealer_pos"] = get_dealer_seat()
        raw_stacks = run_once()
        parsed_stacks, pot_size = self.parse_opponents_data(raw_stacks)
        self.game_state["initial_stacks"] = parsed_stacks
        self.game_state["current_stacks"] = parsed_stacks.copy()
        self.game_state["total_pot"] = pot_size
        self.sync_statuses()
        self.assign_positions()
        self.calculate_math_logic()
        self.display_rich_dashboard()

    def process_turn(self):
        if not self.hand_active: return
        #Wait up to 1 second for the background thread to finish position assignment if needed
        timeout = 0
        while not self.game_state.get("positions") and timeout < 5:
            time.sleep(0.2)
            timeout += 1

        self.ai_advice = None 
        self.ai_response_time = 0.0  # Reset the timer for this turn
        
        # 2A: Get the amount to call from action.py
        button_data = analyze_game()
        if button_data and isinstance(button_data, dict):
            # If it returns {'action': 'call', 'amount': 0.70}
            self.amount_to_call = float(button_data.get("amount", 0.0))
        else:
            self.amount_to_call = 0.0

        self.game_state["board"] = self.sanitize_cards(get_live_board())
        raw_stacks = run_once()
        new_stacks, pot_size = self.parse_opponents_data(raw_stacks)
        if pot_size > 0: self.game_state["total_pot"] = pot_size
        self.sync_statuses()

        if new_stacks:
            for seat, curr_float in new_stacks.items():
                prev_float = self.game_state["current_stacks"].get(seat, 0.0)
                bet_amount = round(prev_float - curr_float, 2)
                if bet_amount > 0:
                    self.game_state["action_history"].append({"seat": seat, "bet": bet_amount, "type": "bet"})
            self.game_state["current_stacks"] = new_stacks

        self.calculate_math_logic() 
        
        # Only spend network/API latency on complex spots. Obvious local folds,
        # checks, and weak-hand decisions stay entirely inside DecisionEngine.
        if not self._should_request_ai_advice():
            self.ai_advice = None
            self.ai_source = "Local Math"
            self.display_rich_dashboard()
            return

        # --- THREADED AI REQUEST FOR COMPLEX SPOTS ---
        # 1. Show the dashboard instantly with Math logic and a "Thinking" placeholder
        self.ai_advice = {"decision": "THINKING...", "thought_process": "Complex spot; requesting network analysis..."}
        self.ai_source = "Network"
        self.display_rich_dashboard()

        # 2. Fire off the AI request in a background thread
        threading.Thread(target=self._fetch_ai_and_update, daemon=True).start()

    def _fetch_ai_and_update(self):
        """Background worker to fetch AI advice without freezing the UI."""
        self.request_ai_advice()
        self.display_rich_dashboard() # Update the dashboard again once the API returns

orchestrator = PokerOrchestrator()

def on_press(key):
    try:
        if hasattr(key, 'char'):
            if key.char == 'a': orchestrator.toggle_automation()
            elif key.char == 'p': orchestrator.start_new_hand()
            elif key.char == 'o': orchestrator.process_turn()
    except Exception: pass
    if key == keyboard.Key.esc: 
        orchestrator.auto_running = False # Cleanly exit the loop
        return False

if __name__ == "__main__":
    with keyboard.Listener(on_press=on_press) as listener: 
        listener.join()