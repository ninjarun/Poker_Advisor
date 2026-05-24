import pyautogui
from pynput import keyboard

# Store our points here temporarily
points = []

def on_press(key):
    try:
        # Detect 'c' key press
        if hasattr(key, 'char') and key.char == 'c':
            x, y = pyautogui.position()
            points.append((x, y))
            
            if len(points) == 1:
                print(f"Top-Left captured: ({x}, {y}) --> Now hover over Bottom-Right and press 'c'.")
            elif len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Output the 4 values perfectly formatted for the scaler script
                print(f"\n✅ Output for script: ({x1}, {y1}, {x2}, {y2})\n")
                
                # Clear the points list so you can immediately capture another box
                points.clear()
                print("-" * 30)
                print("Ready for next capture. Hover over Top-Left and press 'c'.")
                
    except AttributeError:
        pass

    # Stop script when Esc is pressed
    if key == keyboard.Key.esc:
        print("\nCapture finished.")
        return False

print("--- 4-Value Bounding Box Grabber ---")
print("1. Hover mouse over the TOP-LEFT corner of your target.")
print("2. Press 'c' to lock the first point.")
print("3. Hover mouse over the BOTTOM-RIGHT corner.")
print("4. Press 'c' again to lock the second point and get your 4 values.")
print("5. Press 'Esc' to exit.")
print("-" * 40)

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()


# import pyautogui
# from pynput import keyboard

# def on_press(key):
#     try:
#         # Detect 'c' key press
#         if hasattr(key, 'char') and key.char == 'c':
#             x, y = pyautogui.position()
            
#             # Instantly output the (X, Y) coordinate
#             print(f"✅ Captured Point: ({x}, {y})")
                
#     except AttributeError:
#         pass

#     # Stop script when Esc is pressed
#     if key == keyboard.Key.esc:
#         print("\nCapture finished.")
#         return False

# print("--- Single Point (X, Y) Grabber ---")
# print("1. Hover mouse over your target point.")
# print("2. Press 'c' to instantly capture the (X, Y) coordinate.")
# print("3. You can press 'c' as many times as you need for different spots.")
# print("4. Press 'Esc' to exit.")
# print("-" * 35)

# # Start listening to the keyboard
# with keyboard.Listener(on_press=on_press) as listener:
#     listener.join()