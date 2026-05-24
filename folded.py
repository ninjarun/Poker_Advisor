import pyautogui
import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
ACTIVE_TEMPLATE_DIR = "templates/folded"
DEBUG_DIR_FOLDED = "debug_folded"

# Create the debug directory if it doesn't exist
if not os.path.exists(DEBUG_DIR_FOLDED):
    os.makedirs(DEBUG_DIR_FOLDED)

# Make sure these match the output you got from the new fix_coordinates.py!
ANCHOR_POINTS = {
    "Opponent_1": (452, 679),
    "Opponent_2": (496, 268),
    "Opponent_3": (1005, 167),
    "Opponent_4": (1514, 268),
    "Opponent_5": (1558, 681),
}



BOX_W = 45
BOX_H = 25

def check_active_status():
    # 1. Capture Current Screen
    screenshot = pyautogui.screenshot()
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    active_players = []
    timestamp = datetime.now().strftime('%H%M%S')
    print(f"--- Scanning for Active Players ({datetime.now().strftime('%H:%M:%S')}) ---")

    for name, (anchor_x, anchor_y) in ANCHOR_POINTS.items():
        # 2. Crop the current live seat (Up and Left from anchor)
        x1, y1 = anchor_x - BOX_W, anchor_y - BOX_H
        current_roi = img_cv[y1:anchor_y, x1:anchor_x]

        # --- DEBUG SAVE ---
        # Saves exactly what the script is looking at into the debug folder
        debug_filename = f"crop_{name}_{timestamp}.png"
        debug_path = os.path.join(DEBUG_DIR_FOLDED, debug_filename)
        cv2.imwrite(debug_path, current_roi)

        # 3. Load the corresponding "Active" template you saved earlier
        try:
            template_files = [f for f in os.listdir(ACTIVE_TEMPLATE_DIR) if f.startswith(name)]
            if not template_files:
                print(f" [!] Missing template for {name}")
                continue
            
            template_path = os.path.join(ACTIVE_TEMPLATE_DIR, template_files[0])
            template = cv2.imread(template_path)
        except Exception as e:
            print(f" [!] Error loading template for {name}: {e}")
            continue

        # 4. Perform Template Matching
        result = cv2.matchTemplate(current_roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 5. Logic: High Match = ACTIVE
        threshold = 0.45 

        if max_val > threshold:
            status = "ACTIVE"
            active_players.append(name)
        else:
            status = "FOLDED"

        print(f"[{name}]: {status} (Match Score: {max_val:.4f})")

    return active_players

if __name__ == "__main__":
    active = check_active_status()
    print(f"\nFinal Active List: {active}")
    # print(f"\n✅ Debug crops saved. Check the '{DEBUG_DIR_FOLDED}' folder to see if the cards fit in the 45x25 box!")