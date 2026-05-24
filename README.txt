# Poker Advisor

Poker Advisor is a real-time, computer-vision-based poker assistant designed to analyze game states and provide Game Theory Optimal (GTO) and Expected Value (EV) decision support. 

It uses a highly accurate hybrid OCR system with template matching to read table states, combined with a modular decision engine that calculates equity, opponent ranges, and optimal bet sizing. To minimize detection risks, the system is designed for manual execution rather than continuous automated looping.

---

## 🚀 Features

* **Real-Time Table Vision:** Uses OpenCV to detect RGB pixel changes indicating when it is your turn to act.
* **Hybrid OCR Card Recognition:** Utilizes tiny, precisely mapped Regions of Interest (ROIs) and template matching (`/templates`) to achieve 100% accuracy in reading board and hand cards.
* **Modular Decision Engine:** * `equity_engine.py`: Calculates hand strength against perceived opponent ranges.
  * `ev_engine.py`: Computes the Expected Value of different actions.
  * `sizing_engine.py`: Recommends optimal bet/raise sizing.
  * `decision_engine.py`: Aggregates data to output a final recommended action.
* **Opponent Tracking:** Monitors folded players and active opponents via dedicated modules (`folded.py`, `opponents.py`).

---

## 📋 Prerequisites

* **Python 3.8+**
* Requirements installed via `pip install -r requirements.txt`
* A supported poker client running on your machine (can be run via VM).

---

## 🛠️ Installation & Setup

Because poker clients scale differently based on monitor resolution, the system uses strict coordinate mapping defined by top-left and bottom-right points.

### 1. Install Dependencies
Clone the repository and install the required Python libraries:
git clone https://github.com/ninjarun/Poker_Advisor.git
cd Poker_Advisor
pip install -r requirements.txt

### 2. Map the Table Coordinates
Before running the advisor, you must calibrate the software to your screen. The ROIs are stored in a flat dictionary format to ensure rapid, exact processing.

1. **Get the Client Window:** Open your poker client and position it where you want it. Run the following script to capture the master window coordinates:
   python coordinates.py
   *Take note of the coordinates output by this script.*

2. **Generate Element Coordinates:**
   Open `fix_coordinates.py` in your editor and paste the window coordinates you just retrieved into the designated variables. Then, run the script:
   python fix_coordinates.py
   This will automatically calculate the exact Top-Left and Bottom-Right bounding boxes for all in-game elements (board cards, hand cards, opponent locations, and action buttons).

---

## 💻 Usage

To maintain account safety and avoid automated loop detection by poker clients, it is recommended to trigger the script manually when you face a complex decision.

With the table open and coordinates configured, run:
python main.py

The script will:
1. Capture the current screen state.
2. Read your hand (`hand_cards.py`) and the community cards (`board_cards.py`).
3. Determine active vs. folded opponents.
4. Pass the `game_state` through the decision engines.
5. Output the recommended GTO action and bet sizing.

---

## 🏗️ Project Architecture

* **`/engine`**: The core logic. Processes the math, ranges, and expected value calculations.
* **`/models`**: Data structures, including the `game_state.py` object that holds current hand data.
* **`/templates`**: Contains the cropped images of ranks, suits, and folded opponent states used by the vision system for exact template matching.
* **`main.py`**: The entry point that orchestrates the vision capture and engine evaluation.
* **`action.py` / `dealer.py`**: Handles identifying available actions and dealer button positioning.

---

## ⚠️ Disclaimer

This software is for educational and theoretical study of game theory and computer vision. Check the Terms of Service of your specific poker platform regarding the use of real-time assistance (RTA) tools, as using them during live play is strictly prohibited by most major providers.
