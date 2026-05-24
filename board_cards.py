import pyautogui
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

SUIT_SYMBOLS = {
    'h': '♥',
    'd': '♦',
    'c': '♣',
    's': '♠'
}


def identify_template(img_pil, template_dir, threshold=0.60):
    """Matches a crop against templates WITHOUT resizing."""
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    _, target_bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    best_score = -1
    best_label = None
    
    if not os.path.exists(template_dir):
        return None, 0

    for filename in os.listdir(template_dir):
        if not filename.endswith(".png"): continue
        
        # template_gray = cv2.imread(os.path.join(template_dir, filename), 0)
        template_pil = Image.open(os.path.join(template_dir, filename)).convert('RGB')
        template_gray = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2GRAY)
        if template_gray is None: continue
        
        _, template_bw = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        
        # --- RESIZING REMOVED HERE ---
        # Get dimensions to ensure template isn't bigger than the crop
        t_h, t_w = template_bw.shape[:2]
        img_h, img_w = target_bw.shape[:2]

        if t_h > img_h or t_w > img_w:
            # Skip if the template is too big for the crop to avoid OpenCV errors
            continue
        
        res = cv2.matchTemplate(target_bw, template_bw, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = max_val
            # Removes the .png, then ignores anything after a hyphen
            raw_name = filename.split('.')[0]
            best_label = raw_name.split('-')[0]
            
    return (best_label, best_score) if best_score >= threshold else (None, best_score)

# def process_board_image(board_img, source_name):
#     """Core logic to detect cards from a board image object."""
#     bw, bh = board_img.size
    
#     gray_board = cv2.cvtColor(np.array(board_img), cv2.COLOR_RGB2GRAY)
#     _, binary_board = cv2.threshold(gray_board, 200, 255, cv2.THRESH_BINARY)

#     found_cards = []
#     # print(f"\n--- SCANNING BOARD: {source_name} ---")

#     debug_crop_dir = "debug_crops"
#     if not os.path.exists(debug_crop_dir):
#         os.makedirs(debug_crop_dir)

#     for i in range(5):
#         expected_center = int((i + 0.5) * (bw / 5))
        
#         search_zone = binary_board[20, max(0, expected_center-60) : min(bw, expected_center+80)]
#         white_pixels = np.where(search_zone == 255)[0]
        
#         if len(white_pixels) > 0:
#             actual_start_x = max(0, expected_center - 60 + white_pixels[0])
            
#             rank_crop = board_img.crop((actual_start_x - 5, 0, actual_start_x + 55, 50))
#             rank_filename = f"{source_name}_slot_{i}_rank.png"
#             rank_crop.save(os.path.join(debug_crop_dir, rank_filename))
#             rank, r_score = identify_template(rank_crop, "templates/ranks", threshold=0.70)
            
#             suit_crop = board_img.crop((actual_start_x + 20, 60, actual_start_x + 95, 135))
#             # suit_crop = board_img.crop((actual_start_x - 5, 50, actual_start_x + 35, 95))
#             suit_filename = f"{source_name}_slot_{i}_suit.png"
#             suit_crop.save(os.path.join(debug_crop_dir, suit_filename))
#             suit, s_score = identify_template(suit_crop, "templates/suits", threshold=0.70)
            
#             if rank and suit:
#                 display_suit = SUIT_SYMBOLS.get(suit, suit) 
#                 card_string = f"{rank}{display_suit}"
#                 found_cards.append(card_string)
#                 print(f"Slot {i}: {card_string} (R:{r_score:.2f} S:{s_score:.2f})")
    
#             else:
#                 # Restored the debug print so it stops failing silently!
#                 print(f"Slot {i}: Failed match (Rank: {rank} [{r_score:.2f}], Suit: {suit} [{s_score:.2f}])")
#         else:
#             print(f"Slot {i}: [ Empty ]")
            
#     return found_cards
def process_board_image(board_img, prefix="DEBUG"):
    """Processes the board screenshot to identify up to 5 cards."""
    # Convert to grayscale and then binary to find card edges
    gray_board = cv2.cvtColor(np.array(board_img), cv2.COLOR_RGB2GRAY)
    
    # Increased threshold to 230 to ignore table felt and only see white card edges
    _, binary_board = cv2.threshold(gray_board, 230, 255, cv2.THRESH_BINARY)
    
    # Optional: Save this to see exactly what the script "sees" as white
    # cv2.imwrite("debug_binary_check.png", binary_board)

    bw, bh = board_img.size
    found_cards = []

    for i in range(5):
        # Calculate where the card should roughly be
        expected_center = int((i + 0.5) * (bw / 5))
        
        # Widened search zone to the right (+110) to catch the 5th card
        search_start = max(0, expected_center - 60)
        search_end = min(bw, expected_center + 110)
        search_zone = binary_board[20, search_start : search_end]
        
        white_pixels = np.where(search_zone == 255)[0]

        if len(white_pixels) > 0:
            # Calculate actual start based on the first white pixel found in zone
            actual_start_x = search_start + white_pixels[0]
            
            # 1. Rank Crop (Top Left)
            rank_crop = board_img.crop((actual_start_x - 5, 0, actual_start_x + 55, 50))
            
            # Safety check: If the crop is too dark, it's not a real card
            if np.array(rank_crop).mean() < 50:
                continue

            # 2. Suit Crop (Center - slightly wider to be safe)
            suit_crop = board_img.crop((actual_start_x + 15, 60, actual_start_x + 100, 135))

            # Debug: Save crops to verify alignment
            timestamp = datetime.now().strftime("%H%M%S")
            # rank_crop.save(f"debug_boards/LIVE_{timestamp}_slot_{i}_rank.png")
            # suit_crop.save(f"debug_boards/LIVE_{timestamp}_slot_{i}_suit.png")

            # Perform Template Matching
            rank, r_score = identify_template(rank_crop, "templates/ranks", threshold=0.85)
            suit, s_score = identify_template(suit_crop, "templates/suits", threshold=0.85)
            
            if rank and suit:
                display_suit = SUIT_SYMBOLS.get(suit, suit) 
                card_string = f"{rank}{display_suit}"
                found_cards.append(card_string)
                print(f"Slot {i}: {card_string} (R:{r_score:.2f} S:{s_score:.2f})")
            else:
                # Helps identify if thresholding or chips are the issue
                print(f"Slot {i}: Match Failed (Rank: {rank} [{r_score:.2f}], Suit: {suit} [{s_score:.2f}])")
        else:
            # No white pixels found in this 1/5th of the board
            pass
            
    return found_cards

def get_live_board():
    """Captures the board using EXACT pixel dimensions to preserve templates."""
    # We keep your top-left X/Y coordinates, but FORCE the width/height 
    # to be exactly the size your templates were built for (586x138)
    # ax, ay, aw, ah = (666, 985, 586, 138)
    ax, ay, aw, ah =     (669, 411, 578, 142)
    # ax, ay = (753, 1006)
    # aw, ah = (586, 138)


    board_img = pyautogui.screenshot(region=(ax, ay, aw, ah))
    
    # if not os.path.exists("debug_boards"):
        # os.makedirs("debug_boards")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # board_img.save(f"debug_boards/live_{timestamp}.png")
    
    return process_board_image(board_img, f"LIVE_{timestamp}")

if __name__ == "__main__":
    final_hand = get_live_board()
    print("\nFINAL RESULT:", " ".join(final_hand) if final_hand else "No cards detected.")

