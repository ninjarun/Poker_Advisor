import os
from PIL import Image, ImageGrab

class GameCoordinateScaler:
    def __init__(self):
        # ==========================================
        # 1. Reference Data for Hand & Board 
        # ==========================================
        self.ref_window_original = {"x1": 254, "y1": 980, "x2": 1666, "y2": 1949}
        self.ref_hand = {"x1": 862, "y1": 1681, "x2": 1047, "y2": 1782}
        self.ref_board = {"x1": 669, "y1": 1346, "x2": 1247, "y2": 1488}

        # ==========================================
        # 2. Reference Data for Dealers, Opponents, Pot
        # ==========================================
        self.ref_window_recent = {"x1": 254, "y1": 644, "x2": 1667, "y2": 1616}
        
        self.ref_dealers = {
            "Opponent 1": {"x1": 549, "y1": 1268, "x2": 588, "y2": 1304},
            "Opponent 2": {"x1": 548, "y1": 931,  "x2": 590, "y2": 972},
            "Opponent 3": {"x1": 880, "y1": 879,  "x2": 917, "y2": 916},
            "Opponent 4": {"x1": 1328, "y1": 933, "x2": 1371, "y2": 975},
            "Opponent 5": {"x1": 1334, "y1": 1264, "x2": 1367, "y2": 1304},
            "Hero":       {"x1": 851, "y1": 1297, "x2": 889, "y2": 1332}
        }

        self.ref_opponents = {
            "Hero":       {"x1": 878, "y1": 1522, "x2": 1030, "y2": 1558},
            "Opponent 1": {"x1": 334, "y1": 1348, "x2": 474,  "y2": 1378},
            "Opponent 2": {"x1": 377, "y1": 937,  "x2": 519,  "y2": 967},
            "Opponent 3": {"x1": 886, "y1": 837,  "x2": 1030, "y2": 869},
            "Opponent 4": {"x1": 1392, "y1": 937, "x2": 1535, "y2": 971},
            "Opponent 5": {"x1": 1433, "y1": 1348, "x2": 1581, "y2": 1381}
        }

        self.ref_pot = {"x1": 863, "y1": 972, "x2": 1052, "y2": 1002}

        # ==========================================
        # 3. Reference Data for Folded Cards
        # ==========================================
        self.ref_window_folded = {"x1": 256, "y1": 645, "x2": 1667, "y2": 1616}
        self.ref_folded_anchors = {
            "Opponent_1": (454, 1280),
            "Opponent_2": (498, 869),
            "Opponent_3": (1006, 768),
            "Opponent_4": (1515, 869),
            "Opponent_5": (1559, 1282)
        }

        # ==========================================
        # 4. Reference Data for Action Buttons
        # ==========================================
        self.ref_window_actions = {"x1": 254, "y1": 44, "x2": 1667, "y2": 1016}
        self.ref_actions = {
            "left":   {"x1": 1136, "y1": 908, "x2": 1268, "y2": 976},
            "middle": {"x1": 1319, "y1": 907, "x2": 1444, "y2": 978},
            "right":  {"x1": 1501, "y1": 906, "x2": 1628, "y2": 978}
        }

        # ==========================================
        # 5. Calculate all relative ratios
        # ==========================================
        self.hand_ratios = self._calculate_box_ratios(self.ref_hand, self.ref_window_original)
        self.board_ratios = self._calculate_box_ratios(self.ref_board, self.ref_window_original)
        self.pot_ratios = self._calculate_box_ratios(self.ref_pot, self.ref_window_recent)
        
        self.dealer_ratios = {seat: self._calculate_box_ratios(coords, self.ref_window_recent) for seat, coords in self.ref_dealers.items()}
        self.opponent_ratios = {seat: self._calculate_box_ratios(coords, self.ref_window_recent) for seat, coords in self.ref_opponents.items()}
        self.folded_anchor_ratios = {seat: self._calculate_point_ratios(pt, self.ref_window_folded) for seat, pt in self.ref_folded_anchors.items()}
        self.action_ratios = {btn: self._calculate_box_ratios(coords, self.ref_window_actions) for btn, coords in self.ref_actions.items()}

    def _calculate_box_ratios(self, target, ref_win):
        target_width = target["x2"] - target["x1"]
        target_height = target["y2"] - target["y1"]
        ref_w_width = ref_win["x2"] - ref_win["x1"]
        ref_w_height = ref_win["y2"] - ref_win["y1"]

        return {
            "rx": (target["x1"] - ref_win["x1"]) / ref_w_width,
            "ry": (target["y1"] - ref_win["y1"]) / ref_w_height,
            "rw": target_width / ref_w_width,
            "rh": target_height / ref_w_height
        }

    def _calculate_point_ratios(self, pt, ref_win):
        ref_w_width = ref_win["x2"] - ref_win["x1"]
        ref_w_height = ref_win["y2"] - ref_win["y1"]
        return {
            "rx": (pt[0] - ref_win["x1"]) / ref_w_width,
            "ry": (pt[1] - ref_win["y1"]) / ref_w_height
        }

    def get_all_new_coordinates(self, new_win_x1, new_win_y1, new_win_x2, new_win_y2):
        new_width = new_win_x2 - new_win_x1
        new_height = new_win_y2 - new_win_y1

        def apply_box_scale(ratios):
            x = int(new_win_x1 + (new_width * ratios["rx"]))
            y = int(new_win_y1 + (new_height * ratios["ry"]))
            w = int(new_width * ratios["rw"])
            h = int(new_height * ratios["rh"])
            return (x, y, w, h)

        def apply_point_scale(ratios):
            x = int(new_win_x1 + (new_width * ratios["rx"]))
            y = int(new_win_y1 + (new_height * ratios["ry"]))
            return (x, y)

        scaled_hand = apply_box_scale(self.hand_ratios)
        scaled_board = apply_box_scale(self.board_ratios)
        scaled_pot = apply_box_scale(self.pot_ratios)
        scaled_dealers = {seat: apply_box_scale(ratios) for seat, ratios in self.dealer_ratios.items()}
        scaled_opponents = {seat: apply_box_scale(ratios) for seat, ratios in self.opponent_ratios.items()}
        scaled_folded_anchors = {seat: apply_point_scale(ratios) for seat, ratios in self.folded_anchor_ratios.items()}
        scaled_actions = {btn: apply_box_scale(ratios) for btn, ratios in self.action_ratios.items()}

        return scaled_hand, scaled_board, scaled_pot, scaled_dealers, scaled_opponents, scaled_folded_anchors, scaled_actions

    def crop_and_save(self, coords, image_source, output_filename):
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source

        x, y, w, h = coords
        crop_box = (x, y, x + w, y + h)
        
        cropped_img = img.crop(crop_box)
        cropped_img.save(output_filename)
        print(f"✅ Saved test crop: {output_filename}")


if __name__ == "__main__":
    print("=== Poker Coordinate Scaler ===")
    user_input = input("Enter your NEW window coordinates (e.g., 254, 644, 1667, 1616): ")
    
    try:
        clean_input = user_input.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        new_win = tuple(int(x.strip()) for x in clean_input.split(","))
        
        if len(new_win) != 4:
            raise ValueError("You must provide exactly 4 numbers.")
            
    except Exception as e:
        print(f"\n❌ Invalid input format. Error: {e}")
        exit()

    scaler = GameCoordinateScaler()
    scaled_hand, scaled_board, scaled_pot, scaled_dealers, scaled_opponents, scaled_folded_anchors, scaled_actions = scaler.get_all_new_coordinates(*new_win)

    print("\n" + "=" * 50)
    print("⬇️  YOUR NEW HAND & BOARD COORDINATES ⬇️")
    print("=" * 50)
    print(f"Hand:  {scaled_hand}")
    print(f"Board: {scaled_board}\n")

    print("=" * 50)
    print("⬇️  COPY AND PASTE THIS INTO action.py ⬇️")
    print("=" * 50)
    print("BUTTONS = {")
    for btn, coords in scaled_actions.items(): print(f'    "{btn}": {coords},')
    print("}\n")

    print("=" * 50)
    print("⬇️  COPY AND PASTE THIS INTO dealer.py ⬇️")
    print("=" * 50)
    print("DEALER_ROIS = {")
    for seat, coords in scaled_dealers.items(): print(f'    "{seat}": {coords},')
    print("}\n")

    print("=" * 50)
    print("⬇️  COPY AND PASTE THIS INTO opponents.py ⬇️")
    print("=" * 50)
    print("ROIS = {")
    for seat, coords in scaled_opponents.items(): print(f'    "{seat}": {{"stack": {coords}}},')
    print(f'    "Pot": {{"total": {scaled_pot}}}')
    print("}\n")

    print("=" * 50)
    print("⬇️  COPY AND PASTE THIS INTO folded.py ⬇️")
    print("=" * 50)
    print("ANCHOR_POINTS = {")
    for seat, pt in scaled_folded_anchors.items(): print(f'    "{seat}": {pt},')
    print("}\n")

    # Screenshot and crops
    print("Taking a screenshot to verify all crops...")
    screenshot = ImageGrab.grab()
    crops_dir = "crops"
    os.makedirs(crops_dir, exist_ok=True)
    
    scaler.crop_and_save(scaled_hand, screenshot, os.path.join(crops_dir, "hand_crop.png"))
    scaler.crop_and_save(scaled_board, screenshot, os.path.join(crops_dir, "board_crop.png"))
    scaler.crop_and_save(scaled_pot, screenshot, os.path.join(crops_dir, "pot_crop.png"))

    for seat, coords in scaled_dealers.items():
        scaler.crop_and_save(coords, screenshot, os.path.join(crops_dir, f"dealer_{seat.replace(' ', '_').lower()}.png"))
    for seat, coords in scaled_opponents.items():
        scaler.crop_and_save(coords, screenshot, os.path.join(crops_dir, f"stack_{seat.replace(' ', '_').lower()}.png"))
    
    # Action buttons crop verification
    for btn, coords in scaled_actions.items():
        scaler.crop_and_save(coords, screenshot, os.path.join(crops_dir, f"action_{btn}.png"))
    
    # Simulate the folded.py 45x25 anchor box for testing
    for seat, pt in scaled_folded_anchors.items():
        simulated_box = (pt[0]-45, pt[1]-25, 45, 25)
        scaler.crop_and_save(simulated_box, screenshot, os.path.join(crops_dir, f"folded_{seat.lower()}.png"))

    print(f"\n✅ All done! Check the '{crops_dir}' folder to visually verify your boundaries.")