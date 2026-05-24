import cv2
import numpy as np
import pyautogui
import pytesseract
import re
import time
from datetime import datetime
import os

from rich.console import Console
from rich.table import Table

console = Console()

# --- 1. USE ABSOLUTE PATH FOR FOLDER ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DEBUG_DIR_OPPONENTS = os.path.join(BASE_DIR, "debug_opponents")
# if not os.path.exists(DEBUG_DIR_OPPONENTS):
    # os.makedirs(DEBUG_DIR_OPPONENTS)

# Your confirmed, highly accurate coordinates
ROIS = {
    "Hero": {"stack": (878, 922, 152, 36)},
    "Opponent 1": {"stack": (334, 748, 140, 30)},
    "Opponent 2": {"stack": (377, 337, 142, 30)},
    "Opponent 3": {"stack": (886, 237, 144, 32)},
    "Opponent 4": {"stack": (1392, 337, 143, 34)},
    "Opponent 5": {"stack": (1433, 748, 148, 33)},
    "Pot": {"total": (863, 372, 189, 30)}
}



def is_blank(img):
    """Checks if a cropped image is essentially empty based on standard deviation."""
    _, std_dev = cv2.meanStdDev(img)
    return std_dev[0][0] < 5

def clean_numeric(text):
    """Handles thousands, decimals, and identifies 'All-in' states from OCR text."""
    if re.search(r'all[- ]?in', text, re.IGNORECASE):
        return "ALL-IN"

    text = text.replace(' ', '').replace('$', '').replace('S', '5')
    
    if ',' in text:
        if re.search(r',\d{3}($|\D)', text):
            text = text.replace(',', '')
        else:
            text = text.replace(',', '.')
            
    cleaned = re.sub(r'[^0-9.]', '', text)
    return cleaned if cleaned else "-"

def prep_crop(crop, is_num=False):
    """Prepares an image crop for OCR by upscaling and thresholding."""
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if is_blank(gray):
        return None
        
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    
    _, thresh = cv2.threshold(upscaled, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    if is_blank(thresh):
        return None

    h, w = thresh.shape
    # Fixed padding width logic to prevent negative border sizes
    padded = cv2.copyMakeBorder(thresh, 15, 15, 0, max(0, 650 - w), cv2.BORDER_CONSTANT, value=255)
    return padded

def run_once():
    """Performs a single screen capture and OCR pass to extract all stacks and the total pot."""
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    num_crops = []
    num_order = []
    results = {s: "-" for s in ROIS.keys()}
    results["Pot"] = "0.0"

    timestamp = datetime.now().strftime("%H-%M-%S")

    # 1. Prepare Stack Crops
    for seat in ["Hero", "Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5"]:
        s_coords = ROIS[seat]["stack"]
        s_img = frame[s_coords[1]:s_coords[1]+s_coords[3], s_coords[0]:s_coords[0]+s_coords[2]]
        
        # --- EXPLICIT DEBUG SAVING (COMMENTED OUT FOR PRODUCTION) ---
        # save_path = os.path.join(DEBUG_DIR_OPPONENTS, f"{seat.replace(' ', '_')}_{timestamp}.png")
        # if s_img.size > 0:
        #      cv2.imwrite(save_path, s_img)
        
        s_proc = prep_crop(s_img, is_num=True)
        if s_proc is not None:
            num_crops.append(s_proc)
            num_order.append(seat)

    # 2. Prepare Pot Crop
    p_coords = ROIS["Pot"]["total"]
    p_img = frame[p_coords[1]:p_coords[1]+p_coords[3], p_coords[0]:p_coords[0]+p_coords[2]]
    
    # --- POT DEBUG SAVING (COMMENTED OUT FOR PRODUCTION) ---
    # pot_save_path = os.path.join(DEBUG_DIR_OPPONENTS, f"pot_{timestamp}.png")
    # if p_img.size > 0:
    #     cv2.imwrite(pot_save_path, p_img)

    p_proc = prep_crop(p_img, is_num=True)
    if p_proc is not None:
        num_crops.append(p_proc)
        num_order.append("Pot")

    # 3. Single OCR Pass
    if num_crops:
        num_config = "--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.,$alin-ALIN"
        nums_raw = pytesseract.image_to_string(cv2.vconcat(num_crops), config=num_config).split('\n')
        nums_cleaned = [n.strip() for n in nums_raw if n.strip()]
        
        for i, seat in enumerate(num_order):
            if i < len(nums_cleaned):
                results[seat] = clean_numeric(nums_cleaned[i])

    # 4. Console Display
    table = Table(title=f"Poker Session - {datetime.now().strftime('%H:%M:%S')}")
    table.add_column("Seat", style="cyan")
    table.add_column("Stack/Value", style="green")
    for s in ["Hero", "Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5"]:
        table.add_row(s, results[s])
    table.add_section()
    table.add_row("TOTAL POT", f"[yellow]{results['Pot']}[/yellow]")
    console.print(table)

    # 5. Data Formatting for main.py
    pot_clean = re.sub(r'[^0-9.]', '', str(results["Pot"]))
    pot_value = float(pot_clean) if pot_clean else 0.0

    formatted_opponents = []
    for seat in ["Hero", "Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5"]:
        clean_val = re.sub(r'[^0-9.]', '', str(results.get(seat, "0.0")))
        formatted_opponents.append({
            "seat": seat,
            "stack": float(clean_val) if clean_val else 0.0
        })

    return {
        "opponents": formatted_opponents,
        "total_pot": pot_value
    }

# ==========================================
# Optional Execution Block (Disabled for Production)
# ==========================================
# if __name__ == "__main__":
#     print("Running a single manual test...")
#     data = run_once()
#     print(data)