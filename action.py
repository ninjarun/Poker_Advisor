import pyautogui
import pytesseract
import os
import time
import re
from datetime import datetime
from PIL import Image

# --- CONFIGURATION ---
# These coordinates are mapped to the buttons on your Ubuntu display (1920x1080)
# BUTTONS = {
#     "left":   (1121, 1475, 172, 85),  # Fold / Check-Fold
#     "middle": (1308, 1475, 172, 85),  # Call / Check
#     "right":  (1495, 1475, 172, 85)   # Raise / Bet
# }
BUTTONS = {
    "left": (1136, 908, 132, 68),
    "middle": (1319, 907, 125, 71),
    "right": (1501, 906, 127, 72),
}


# Color Detection: Looks for the specific Red of active buttons
TARGET_RED = (180, 40, 40) 
TOLERANCE = 80             

def is_button_active(region):
    """Checks if the center of the button is Red (Active) or Black (Inactive)."""
    center_x = region[0] + (region[2] // 2)
    center_y = region[1] + (region[3] // 2)
    try:
        # Standard PyAutoGUI pixel check
        return pyautogui.pixelMatchesColor(center_x, center_y, TARGET_RED, tolerance=TOLERANCE)
    except:
        return False

def clean_poker_text(raw_text):
    """Extracts both the action name and the numerical value from OCR string."""
    text = raw_text.lower().replace('\n', ' ').strip()
    
    action = "unknown"
    if "fold" in text: action = "fold"
    elif "check" in text: action = "check"
    elif "call" in text: action = "call"
    elif any(p in text for p in ["rais", "art", "bet", "ery"]): action = "raise"

    amount_match = re.search(r"(\d+\.\d{2})", text)
    amount = float(amount_match.group(1)) if amount_match else 0.0

    return {"action": action, "amount": amount}

def get_button_data(name, region):
    """Captures, enhances, and OCRs the button text if active."""
    if not is_button_active(region):
        return None

    # 1. Capture and Convert to Grayscale
    img = pyautogui.screenshot(region=region)
    img = img.convert('L')
    
    # 2. Upscale for better OCR accuracy on ThinkPad screen
    w, h = img.size
    img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
    
    # 3. Thresholding for high contrast
    img = img.point(lambda x: 255 if x < 120 else 0, '1')
    
    # 4. OCR Processing
    custom_config = r'--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.$'
    raw_text = pytesseract.image_to_string(img, config=custom_config)
    
    return clean_poker_text(raw_text)

# def analyze_game():
#     """
#     PERFORMS ONE SCAN ONLY.
#     Returns True if buttons are detected (it's your turn), False otherwise.
#     This allows main.py to control the loop and move to Phase 5.
#     """
#     active_count = 0
    
#     for name, roi in BUTTONS.items():
#         if is_button_active(roi):
#             active_count += 1
            
#     # If any red buttons are found, return True to trigger the next phase
#     if active_count > 0:
#         return True
    
#     return False

def analyze_game():
    """
    PERFORMS ONE SCAN.
    Returns the data from the 'middle' button (usually Call/Check) 
    if buttons are detected, otherwise returns None.
    """
    # Check if it's our turn by looking for any active red buttons
    is_turn = False
    for name, roi in BUTTONS.items():
        if is_button_active(roi):
            is_turn = True
            break
            
    if is_turn:
        # Specifically OCR the middle button to get the 'Call' amount
        middle_data = get_button_data("middle", BUTTONS["middle"])
        return middle_data # Returns {'action': 'call', 'amount': 0.70} or similar
    
    return None

def is_my_turn():
    """
    Ultra-fast pixel check ONLY. No OCR. 
    Used for high-speed polling in the main loop to prevent CPU bottlenecks.
    """
    for name, roi in BUTTONS.items():
        if is_button_active(roi):
            return True
    return False

if __name__ == "__main__":
    # If run directly for testing, it will just do one check
    print(f"Checking for turn... Result: {analyze_game()}")