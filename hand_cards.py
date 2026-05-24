
import cv2
import numpy as np
from PIL import ImageGrab
import os
from datetime import datetime

# --- CONFIGURATION ---
# HAND_ROI = (863, 1328, 191, 83) 
HAND_ROI = (862, 747, 185, 101)

RANK_TEMPLATE_DIR = "templates/ranks"
SUIT_TEMPLATE_DIR = "templates/suits"
# DEBUG_DIR = "debug_hand" 
THRESHOLD = 0.70

# Mapping dictionary for visual symbols
SUIT_SYMBOLS = {
    "h": "♥",  # Hearts
    "d": "♦",  # Diamonds
    "s": "♠",  # Spades
    "c": "♣"   # Clubs
}

# if not os.path.exists(DEBUG_DIR):
    # os.makedirs(DEBUG_DIR)

def analyze_rank_structure(roi):
    """
    Structural Tie-Breaker: Checks if the digit has 1 or 2 holes.
    8 has two holes (loops), 9 has one.
    """
    # 1. Binarize the ROI
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure background is black and digit is white for contouring
    # Check corners; if light, invert.
    if np.mean([thresh[0,0], thresh[0,-1], thresh[-1,0], thresh[-1,-1]]) > 127:
        thresh = cv2.bitwise_not(thresh)

    # 2. Count Holes using Hierarchy
    # RETR_CCOMP retrieves all contours and organizes them into a two-level hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    holes = 0
    if hierarchy is not None:
        for i in range(len(contours)):
            # If the contour has a parent, it's an internal hole
            if hierarchy[0][i][3] != -1:
                holes += 1
    
    # 3. Decision Logic
    if holes >= 2:
        return "8"
    if holes == 1:
        return "9"
    
    # 4. Fallback: Density Check (if holes are too small/blurry to detect)
    # Check bottom-left quadrant (approx 30% of width/height)
    h, w = thresh.shape
    bl_sample = thresh[int(h*0.7):h, 0:int(w*0.3)]
    density = cv2.countNonZero(bl_sample) / (bl_sample.size + 1e-6)
    
    return "8" if density > 0.15 else "9"

def identify_component(cv_image, template_folder, is_rank=False):
    """
    Standard matching with a structural override for ranks 8 and 9.
    """
    target_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    best_val = -1
    best_name = ""
    best_loc = (0, 0)
    best_size = (0, 0)

    if not os.path.exists(template_folder):
        return ""

    for filename in os.listdir(template_folder):
        if not filename.lower().endswith('.png'):
            continue
            
        template = cv2.imread(os.path.join(template_folder, filename), 0)
        if template is None:
            continue

        for scale in np.linspace(0.2, 1.2, 30):
            w = int(template.shape[1] * scale)
            h = int(template.shape[0] * scale)
            
            if w > target_gray.shape[1] or h > target_gray.shape[0] or w < 5:
                continue
                
            resized_tpl = cv2.resize(template, (w, h))
            res = cv2.matchTemplate(target_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_name = filename.split('.')[0]
                best_loc = max_loc
                best_size = (w, h)

    # --- NEW: Structural Override for 8/9 confusion ---
    if is_rank and (best_name == "8" or best_name == "9") and best_val > THRESHOLD:
        x, y = best_loc
        w, h = best_size
        # Extract the specific area where the match occurred
        roi = target_gray[y:y+h, x:x+w]
        if roi.size > 0:
            best_name = analyze_rank_structure(roi)

    return best_name if best_val > THRESHOLD else ""

def get_my_hand():
    x, y, w, h = HAND_ROI
    full_hand_cap = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    full_hand_cv = cv2.cvtColor(np.array(full_hand_cap), cv2.COLOR_RGB2BGR)

    card1_img = full_hand_cv[:, 0:70]   
    card2_img = full_hand_cv[:, 80:150] 

    timestamp = datetime.now().strftime("%H-%M-%S")
    current_hand = []
    
    for i, img in enumerate([card1_img, card2_img]):
        # save_path = os.path.join(DEBUG_DIR, f"card_{i+1}_{timestamp}.png")
        # cv2.imwrite(save_path, img)
        
        # We pass is_rank=True so the script knows when to double-check for 8s and 9s
        rank = identify_component(img, RANK_TEMPLATE_DIR, is_rank=True)
        suit_key = identify_component(img, SUIT_TEMPLATE_DIR, is_rank=False)
        
        if rank and suit_key:
            suit_display = SUIT_SYMBOLS.get(suit_key.lower(), suit_key)
            current_hand.append(f"{rank}{suit_display}")
        else:
            current_hand.append("??")

    return current_hand

if __name__ == "__main__":
    hand = get_my_hand()
    print(f"Hand Detected: {', '.join(hand)}")


