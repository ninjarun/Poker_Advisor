import pyautogui
import numpy as np
import cv2
import os
from datetime import datetime

# --- CONFIGURATION ---
# DEBUG_DIR = "debug_dealer"
# if not os.path.exists(DEBUG_DIR):
    # os.makedirs(DEBUG_DIR)

# Using your confirmed point pairs
# DEALER_ROIS = {
#     "Opponent 1": (551, 1241, 586 - 551, 1277 - 1241),
#     "Opponent 4": (1330, 907, 1368 - 1330, 949 - 907),
#     "Opponent 2": (554, 912, 585 - 554, 947 - 912),
#     "Opponent 5": (1331, 1237, 1374 - 1331, 1283 - 1237),
#     "Hero":       (852, 1273, 886 - 852, 1303 - 1273),
#     "Opponent 3": (873, 856, 914 - 873, 892 - 856)
# }
DEALER_ROIS = {
    "Opponent 1": (549, 668, 39, 36),
    "Opponent 2": (548, 331, 42, 41),
    "Opponent 3": (880, 279, 37, 37),
    "Opponent 4": (1328, 333, 43, 42),
    "Opponent 5": (1334, 664, 33, 40),
    "Hero": (851, 697, 38, 35),
}






def get_dealer_seat():
    screenshot = pyautogui.screenshot()
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%H%M%S")
    
    found_seat = "Unknown"
    max_yellow_pixels = 0

    for seat, (x, y, w, h) in DEALER_ROIS.items():
        # 1. Extract Crop
        roi = img_cv[y:y+h, x:x+w]
        
        # 2. Save for verification
        # cv2.imwrite(os.path.join(DEBUG_DIR, f"{timestamp}_{seat}.png"), roi)
        
        # 3. Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 4. Define Yellow Range
        # Lower: Light yellow/mustard | Upper: Bright neon yellow
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # 5. Create a mask and count yellow pixels
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_count = np.sum(mask > 0)
        
        # Debug print to see what the script "sees"
        if yellow_count > 10: # If at least 10 pixels are yellow
            print(f"DEBUG: {seat} has {yellow_count} yellow pixels.")
            
            # The seat with the MOST yellow pixels is the winner
            if yellow_count > max_yellow_pixels:
                max_yellow_pixels = yellow_count
                found_seat = seat
                
    return found_seat

if __name__ == "__main__":
    result = get_dealer_seat()
    print(f"\n---> THE DEALER IS: {result}")


# import pyautogui
# import numpy as np
# import cv2
# import os
# from datetime import datetime

# # --- CONFIGURATION ---
# # Debug directory logic disabled to stop disk writes
# DEBUG_DIR = "debug_dealer"
# # if not os.path.exists(DEBUG_DIR):
# #     os.makedirs(DEBUG_DIR)

# DEALER_ROIS = {
#     "Opponent 1": (551, 1241, 586 - 551, 1277 - 1241),
#     "Opponent 4": (1330, 907, 1368 - 1330, 949 - 907),
#     "Opponent 2": (554, 912, 585 - 554, 947 - 912),
#     "Opponent 5": (1331, 1237, 1374 - 1331, 1283 - 1237),
#     "Hero":       (852, 1273, 886 - 852, 1303 - 1273),
#     "Opponent 3": (873, 856, 914 - 873, 892 - 856)
# }

# def get_dealer_seat():
#     screenshot = pyautogui.screenshot()
#     img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
#     # timestamp = datetime.now().strftime("%H%M%S") # No longer needed for filenames
    
#     found_seat = "Unknown"
#     max_yellow_pixels = 0

#     for seat, (x, y, w, h) in DEALER_ROIS.items():
#         # 1. Extract Crop (Keep in memory for processing)
#         roi = img_cv[y:y+h, x:x+w]
        
#         # 2. SAVE TO DISK REMOVED
#         # cv2.imwrite(os.path.join(DEBUG_DIR, f"{timestamp}_{seat}.png"), roi)
        
#         # 3. Convert to HSV
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
#         # 4. Define Yellow Range
#         lower_yellow = np.array([20, 100, 100])
#         upper_yellow = np.array([35, 255, 255])
        
#         # 5. Create a mask and count yellow pixels
#         mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
#         yellow_count = np.sum(mask > 0)
        
#         if yellow_count > 10: 
#             # We keep the print statement so you can still see logic in the console
#             print(f"DEBUG: {seat} has {yellow_count} yellow pixels.")
            
#             if yellow_count > max_yellow_pixels:
#                 max_yellow_pixels = yellow_count
#                 found_seat = seat
                
#     return found_seat

# if __name__ == "__main__":
#     result = get_dealer_seat()
#     print(f"\n---> THE DEALER IS: {result}")