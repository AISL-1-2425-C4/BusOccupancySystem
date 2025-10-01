"""
Seating processor - extracted from seating.py as a reusable function
Contains the full complex algorithm for processing detection results into bus seating layout
"""

import json
# import numpy as np  # Removed for Vercel compatibility

def process_seating_layout(detections_input):
    """
    Process detection results into seating layout using the full seating.py algorithm
    
    Args:
        detections_input: List of detection dictionaries with x_min, y_min, x_max, y_max, class_id
    
    Returns:
        Dictionary with row-based seating layout, or None if processing fails
    """
    detections = detections_input
    print(f"🚌 Processing {len(detections)} detections using full seating.py algorithm...")

    if not detections:
        return None

    # All helper functions and logic as before...
    try:
        # Extract midpoints from detections
        midpoints = []
        for detection in detections:
            x_min, y_min = detection["x_min"], detection["y_min"]
            x_max, y_max = detection["x_max"], detection["y_max"]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            midpoints.append((x_center, y_center))

        # Sort midpoints by y (top to bottom)
        midpoints.sort(key=lambda pt: pt[1])

        # Group midpoints into rows by y proximity
        row_threshold = 60  # pixels, adjust if needed for your image
        rows = []
        for pt in midpoints:
            placed = False
            for row in rows:
                if abs(row[0][1] - pt[1]) < row_threshold:
                    row.append(pt)
                    placed = True
                    break
            if not placed:
                rows.append([pt])

        # Sort each row by x (left to right)
        for row in rows:
            row.sort(key=lambda pt: pt[0])

        # Improved aisle detection - iteratively adjust until balanced (disparity ≤ 1)
        x_coords = [pt[0] for pt in midpoints]
        x_min, x_max = min(x_coords), max(x_coords)
        total_seats = len(x_coords)
        target_left = total_seats // 2  # Target roughly half the seats on each side

        print(f"Total seats: {total_seats}, Target left side: {target_left}")

        # Use pure Python to test aisle positions (Vercel compatible)
        start_pos = x_min + (x_max - x_min) * 0.1
        end_pos = x_min + (x_max - x_min) * 0.9
        step = (end_pos - start_pos) / 199  # 200 points = 199 intervals
        test_positions = [start_pos + i * step for i in range(200)]
        
        best_balance = float('inf')
        best_aisle_pos = x_min + (x_max - x_min) / 2
        
        for test_pos in test_positions:
            left_count = sum(1 for x in x_coords if x < test_pos)
            right_count = sum(1 for x in x_coords if x >= test_pos)
            
            balance_score = abs(left_count - right_count)
            
            if balance_score < best_balance:
                best_balance = balance_score
                best_aisle_pos = test_pos
                
                if balance_score <= 1:
                    break

        aisle_position = best_aisle_pos
        print(f"Optimal aisle position: {aisle_position:.1f}")

        # Separate seats into left and right groups
        left_seats = [pt for pt in midpoints if pt[0] < aisle_position]
        right_seats = [pt for pt in midpoints if pt[0] >= aisle_position]

        print(f"Left side seats: {len(left_seats)}, Right side seats: {len(right_seats)}")

        # Find cross-aisle pair (closest seats across the aisle)
        aisle_pair = None
        if left_seats and right_seats:
            min_distance = float('inf')
            for left_pt in left_seats:
                for right_pt in right_seats:
                    # Check if they're roughly at the same Y level (same row)
                    y_diff = abs(left_pt[1] - right_pt[1])
                    if y_diff < 30:  # Same row threshold
                        distance = abs(left_pt[0] - right_pt[0])
                        if distance < min_distance:
                            min_distance = distance
                            aisle_pair = (left_pt, right_pt)

        print(f"Cross-aisle pair found: {aisle_pair is not None}")

        # Find pairs within each side based on closest Y-coordinates
        def find_side_pairs(seats, side_name, exclude_seats=None):
            # After all pairing, handle unpaired dots by generating virtual dots for pairing
            # Only run this after all other passes
            def get_side_of_dot(dot, pairs):
                # Find the closest paired dot and its side (left or right)
                min_dist = float('inf')
                side = None
                for a, b in pairs:
                    for paired_dot in (a, b):
                        dist = abs(dot[0] - paired_dot[0])
                        if dist < min_dist:
                            min_dist = dist
                            side = 'right' if dot[0] > paired_dot[0] else 'left'
                return side

            # Track generated virtual dots for later labeling
            generated_virtual_dots = []

            # Helper to find if a seat is in any pair
            def find_pair_for(seat, pairs):
                for idx, (a, b) in enumerate(pairs):
                    if seat == a or seat == b:
                        return idx, (a, b)
                return None, None
            """Custom pairing: left pairs by lowest y, right pairs by lowest x, both with y-threshold."""
            if len(seats) < 2:
                return []
            if exclude_seats is None:
                exclude_seats = []
            available_seats = [seat for seat in seats if seat not in exclude_seats]
            if len(available_seats) < 2:
                print(f"{side_name} side: Not enough available seats for pairing (have {len(available_seats)}, need 2+)")
                return []
            pairs = []
            y_threshold = 35  # Slightly increased threshold for right side
            x_threshold = 50 if side_name.lower() == "left" else 30  # Stricter for left side
            if side_name.lower() == "left" or side_name.lower() == "right":
                # Sort from bottom to top (highest y to lowest y)
                sorted_seats = sorted(available_seats, key=lambda pt: pt[1], reverse=True)
                paired = set()
                # Always pair the topmost (least y) two seats first (last row), if possible
                if len(sorted_seats) >= 2:
                    s1, s2 = sorted_seats[-2], sorted_seats[-1]
                    y_diff = abs(s1[1] - s2[1])
                    x_diff = abs(s1[0] - s2[0])
                    if y_diff <= y_threshold and x_diff > x_threshold:
                        pairs.append((s1, s2))
                        paired.add(s1)
                        paired.add(s2)
                        print(f"[{side_name.capitalize()} Last Row Pairing] Paired {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                        # Remove these from the list
                        sorted_seats = sorted_seats[:-2]
                    else:
                        print(f"[{side_name.capitalize()} Last Row Pairing] Skipped pairing {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                # Now pair the rest from bottom to top (excluding aisle pair)
                i = 0
                while i + 1 < len(sorted_seats):
                    s1, s2 = sorted_seats[i], sorted_seats[i+1]
                    if s1 in paired or s2 in paired:
                        i += 1
                        continue
                    y_diff = abs(s1[1] - s2[1])
                    x_diff = abs(s1[0] - s2[0])
                    if y_diff <= y_threshold and x_diff > x_threshold:
                        pairs.append((s1, s2))
                        paired.add(s1)
                        paired.add(s2)
                        print(f"[{side_name.capitalize()} Pairing] Paired {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                        i += 2
                    else:
                        print(f"[{side_name.capitalize()} Pairing] Skipped pairing {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                        i += 1
                # Second pass: for all remaining unpaired seats, always try to pair with the next available seat below within threshold
                unpaired = [s for s in sorted_seats if s not in paired]
                i = 0
                while i + 1 < len(unpaired):
                    s1, s2 = unpaired[i], unpaired[i+1]
                    y_diff = abs(s1[1] - s2[1])
                    x_diff = abs(s1[0] - s2[0])
                    if y_diff <= y_threshold and x_diff > x_threshold:
                        pairs.append((s1, s2))
                        paired.add(s1)
                        paired.add(s2)
                        print(f"[{side_name.capitalize()} Second Pass] Paired {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                        i += 2
                    else:
                        print(f"[{side_name.capitalize()} Second Pass] Skipped pairing {s1} and {s2} (y_diff={y_diff:.1f}, x_diff={x_diff:.1f})")
                        i += 1

                # Third pass: re-pairing for optimal y-difference
                # For each dot not in last row, check if a better (closer) partner exists
                all_seats = sorted_seats + ([s1, s2] if len(sorted_seats) < len(available_seats) else [])
                for s in all_seats:
                    if s in paired:
                        continue
                    # Find all other seats in group within y-threshold
                    candidates = [(other, abs(s[1] - other[1])) for other in all_seats if other != s and abs(s[1] - other[1]) <= y_threshold]
                    if not candidates:
                        continue
                    # Find the closest candidate
                    best, best_y = min(candidates, key=lambda x: x[1])
                    # If best is already paired, check if its current pair is a worse match
                    idx, current_pair = find_pair_for(best, pairs)
                    if current_pair:
                        # Find y-diff of current pair
                        other_in_pair = current_pair[0] if current_pair[1] == best else current_pair[1]
                        current_y = abs(best[1] - other_in_pair[1])
                        if best_y < current_y:
                            # Unpair the worse match and re-pair with the closer one
                            pairs.pop(idx)
                            paired.discard(other_in_pair)
                            pairs.append((s, best))
                            paired.add(s)
                            paired.add(best)
                            print(f"[{side_name.capitalize()} Re-Pairing] {s} re-paired with {best} (y_diff={best_y:.1f}), replacing previous pair (y_diff={current_y:.1f})")
            else:
                print(f"[Pairing] Unknown side: {side_name}")
            print(f"{side_name} side pairs: {len(pairs)} (excluded {len(exclude_seats)} cross-aisle seats)")
            # Find all unpaired dots after all passes
            all_paired = set()
            for a, b in pairs:
                all_paired.add(a)
                all_paired.add(b)
            unpaired_final = [s for s in available_seats if s not in all_paired]
            for dot in unpaired_final:
                # Determine which side to generate the virtual dot
                side = get_side_of_dot(dot, pairs)
                # Offset for virtual dot (user requested: x +/- 50, y same)
                offset = 50
                if side == 'right':
                    # Generate to the left
                    virtual_dot = (dot[0] - offset, dot[1])
                else:
                    # Generate to the right
                    virtual_dot = (dot[0] + offset, dot[1])
                pairs.append((dot, virtual_dot))
                generated_virtual_dots.append(virtual_dot)
                print(f"[Virtual Pairing] Paired {dot} with generated dot {virtual_dot} (side: {side}, offset: {offset})")
            # Attach generated_virtual_dots to the function for later use
            find_side_pairs.generated_virtual_dots = generated_virtual_dots
            return pairs

        # Get the seats that are already in the cross-aisle pair to exclude them
        cross_aisle_seats = []
        if aisle_pair:
            cross_aisle_seats = [aisle_pair[0], aisle_pair[1]]

        left_side_pairs = find_side_pairs(left_seats, "Left", exclude_seats=[aisle_pair[0]] if aisle_pair else [])
        right_side_pairs = find_side_pairs(right_seats, "Right", exclude_seats=[aisle_pair[1]] if aisle_pair else [])

        print(f"Left side pairs: {len(left_side_pairs)}, Right side pairs: {len(right_side_pairs)}")

        # Pair LEFT pairs with RIGHT pairs (cross-aisle pair pairing)
        def pair_left_with_right_pairs(left_pairs, right_pairs):
            """Pair left side pairs with right side pairs based on Y-coordinate proximity"""
            if not left_pairs or not right_pairs:
                return []
            
            # Calculate center Y-coordinate for each pair
            left_pair_centers = []
            for pair in left_pairs:
                center_y = (pair[0][1] + pair[1][1]) / 2
                left_pair_centers.append((pair, center_y))
            
            right_pair_centers = []
            for pair in right_pairs:
                center_y = (pair[0][1] + pair[1][1]) / 2
                right_pair_centers.append((pair, center_y))
            
            # Sort pairs by their Y-coordinate (bottom to top)
            left_pair_centers_sorted = sorted(left_pair_centers, key=lambda x: x[1], reverse=True)
            right_pair_centers_sorted = sorted(right_pair_centers, key=lambda x: x[1], reverse=True)
            
            cross_aisle_pair_groups = []
            used_left = set()
            used_right = set()
            
            # Process pairs to create cross-aisle groups
            for i, (left_pair, left_y) in enumerate(left_pair_centers_sorted):
                if i in used_left:
                    continue
                
                best_right_pair = None
                best_right_idx = None
                min_y_distance = float('inf')
                
                for j, (right_pair, right_y) in enumerate(right_pair_centers_sorted):
                    if j in used_right:
                        continue
                    
                    y_distance = abs(left_y - right_y)
                    if y_distance < min_y_distance:
                        min_y_distance = y_distance
                        best_right_pair = right_pair
                        best_right_idx = j
                
                if best_right_pair is not None:
                    cross_aisle_pair_groups.append({
                        'left_pair': left_pair,
                        'right_pair': best_right_pair,
                        'y_center': (left_y + right_pair_centers_sorted[best_right_idx][1]) / 2
                    })
                    used_left.add(i)
                    used_right.add(best_right_idx)
            
            return cross_aisle_pair_groups

        cross_aisle_pair_groups = pair_left_with_right_pairs(left_side_pairs, right_side_pairs)
        print(f"Cross-aisle pair groups: {len(cross_aisle_pair_groups)}")

        # Create function to find class_id from coordinates
        def find_class_id_by_coordinates(x, y, detections, tolerance=50):
            if hasattr(find_side_pairs, 'generated_virtual_dots'):
                for vdot in find_side_pairs.generated_virtual_dots:
                    if abs(vdot[0] - x) < 1e-3 and abs(vdot[1] - y) < 1e-3:
                        return 2  # Unknown / generated
            for detection in detections:
                det_x_center = (detection["x_min"] + detection["x_max"]) / 2
                det_y_center = (detection["y_min"] + detection["y_max"]) / 2
                if abs(det_x_center - x) < tolerance and abs(det_y_center - y) < tolerance:
                    return detection["class_id"]
            return 2  # 👈 instead of None or 1


        # --- Begin: seating.py-mimic last row (6-seat group) logic ---
        def generate_virtual_pair_for(pair, side):
            # Generate a virtual pair on the opposite side for the given pair
            # For last row: generate a pair of class_id 2 seats on the opposite side, 100 and 150 pixels from the aisle, same y as the real pair
            y1 = pair[0][1]
            y2 = pair[1][1]
            if side == 'left':
                # Generate on right side
                x1 = aisle_position + 100
                x2 = aisle_position + 150
            else:
                # Generate on left side
                x1 = aisle_position - 100
                x2 = aisle_position - 150
            return ((x1, y1), (x2, y2))


        last_row_group = None
        if left_side_pairs and right_side_pairs:
            left_lowest_pair = min(left_side_pairs, key=lambda pair: (pair[0][1] + pair[1][1]) / 2)
            right_lowest_pair = min(right_side_pairs, key=lambda pair: (pair[0][1] + pair[1][1]) / 2)

            # Determine last row group and cross-aisle pair (real or virtual)
            if aisle_pair:
                cross_aisle = aisle_pair
                last_row_group = {
                    'bottom_left_pair': left_lowest_pair,
                    'cross_aisle_pair': cross_aisle,
                    'bottom_right_pair': right_lowest_pair,
                    'total_chairs': 6
                }
            else:
                left_y = (left_lowest_pair[0][1] + left_lowest_pair[1][1]) / 2
                right_y = (right_lowest_pair[0][1] + right_lowest_pair[1][1]) / 2
                if left_y > right_y:
                    cross_aisle = generate_virtual_pair_for(left_lowest_pair, 'left')
                    last_row_group = {
                        'bottom_left_pair': left_lowest_pair,
                        'cross_aisle_pair': cross_aisle,
                        'bottom_right_pair': right_lowest_pair,
                        'total_chairs': 6
                    }
                else:
                    cross_aisle = generate_virtual_pair_for(right_lowest_pair, 'right')
                    last_row_group = {
                        'bottom_left_pair': left_lowest_pair,
                        'cross_aisle_pair': cross_aisle,
                        'bottom_right_pair': right_lowest_pair,
                        'total_chairs': 6
                    }
            # Remove any cross-aisle pair group that matches the last row's left and right pairs
            to_remove = []
            for group in cross_aisle_pair_groups:
                if (group['left_pair'] == left_lowest_pair and group['right_pair'] == right_lowest_pair):
                    to_remove.append(group)
            for group in to_remove:
                cross_aisle_pair_groups.remove(group)
        else:
            last_row_group = None

        # Create row-based JSON output (with last row group)
        def create_row_json_output_with_last_row(cross_aisle_pair_groups, aisle_pair, detections, last_row_group):
            """Create JSON output organized by rows (bottom to top) with class_ids, including last row group if present"""
            # Sort cross-aisle pair groups by Y-coordinate (bottom to top)
            sorted_groups = sorted(cross_aisle_pair_groups, 
                                  key=lambda group: group['y_center'], 
                                  reverse=True)
            row_data = {}
            regular_row_idx = 1
            # Process regular rows (excluding the one that's part of the last row)
            for group in sorted_groups:
                row_name = f"row_{regular_row_idx}"
                left_seat1 = group['left_pair'][0]
                left_seat2 = group['left_pair'][1]
                right_seat1 = group['right_pair'][0]
                right_seat2 = group['right_pair'][1]
                left_seats_sorted = sorted([left_seat1, left_seat2], key=lambda seat: seat[0])
                right_seats_sorted = sorted([right_seat1, right_seat2], key=lambda seat: seat[0])
                left_class_id_1 = find_class_id_by_coordinates(left_seats_sorted[0][0], left_seats_sorted[0][1], detections)
                left_class_id_2 = find_class_id_by_coordinates(left_seats_sorted[1][0], left_seats_sorted[1][1], detections)
                right_class_id_1 = find_class_id_by_coordinates(right_seats_sorted[0][0], right_seats_sorted[0][1], detections)
                right_class_id_2 = find_class_id_by_coordinates(right_seats_sorted[1][0], right_seats_sorted[1][1], detections)
                row_data[row_name] = {
                    "column_one": {
                        "class_id": left_class_id_1,
                        "coordinates": {"x": left_seats_sorted[0][0], "y": left_seats_sorted[0][1]}
                    },
                    "column_two": {
                        "class_id": left_class_id_2,
                        "coordinates": {"x": left_seats_sorted[1][0], "y": left_seats_sorted[1][1]}
                    },
                    "column_three": {
                        "class_id": right_class_id_1,
                        "coordinates": {"x": right_seats_sorted[0][0], "y": right_seats_sorted[0][1]}
                    },
                    "column_four": {
                        "class_id": right_class_id_2,
                        "coordinates": {"x": right_seats_sorted[1][0], "y": right_seats_sorted[1][1]}
                    }
                }
                print(f"Row {regular_row_idx}: Left({left_class_id_1},{left_class_id_2}) Right({right_class_id_1},{right_class_id_2})")
                regular_row_idx += 1
            # Add the special LAST ROW with 6 seats (if present)
            if last_row_group and aisle_pair:
                left_pair = last_row_group['bottom_left_pair']
                cross_pair = last_row_group['cross_aisle_pair']
                right_pair = last_row_group['bottom_right_pair']
                left_seats_sorted = sorted([left_pair[0], left_pair[1]], key=lambda seat: seat[0])
                right_seats_sorted = sorted([right_pair[0], right_pair[1]], key=lambda seat: seat[0])
                cross_seats_sorted = sorted([cross_pair[0], cross_pair[1]], key=lambda seat: seat[0])
                left_class_id_1 = find_class_id_by_coordinates(left_seats_sorted[0][0], left_seats_sorted[0][1], detections)
                left_class_id_2 = find_class_id_by_coordinates(left_seats_sorted[1][0], left_seats_sorted[1][1], detections)
                cross_class_id_1 = find_class_id_by_coordinates(cross_seats_sorted[0][0], cross_seats_sorted[0][1], detections)
                cross_class_id_2 = find_class_id_by_coordinates(cross_seats_sorted[1][0], cross_seats_sorted[1][1], detections)
                right_class_id_1 = find_class_id_by_coordinates(right_seats_sorted[0][0], right_seats_sorted[0][1], detections)
                right_class_id_2 = find_class_id_by_coordinates(right_seats_sorted[1][0], right_seats_sorted[1][1], detections)
                row_data['last_row'] = {
                    "column_one": {
                        "class_id": left_class_id_1,
                        "coordinates": {"x": left_seats_sorted[0][0], "y": left_seats_sorted[0][1]}
                    },
                    "column_two": {
                        "class_id": left_class_id_2,
                        "coordinates": {"x": left_seats_sorted[1][0], "y": left_seats_sorted[1][1]}
                    },
                    "column_three": {
                        "class_id": cross_class_id_1,
                        "coordinates": {"x": cross_seats_sorted[0][0], "y": cross_seats_sorted[0][1]}
                    },
                    "column_four": {
                        "class_id": cross_class_id_2,
                        "coordinates": {"x": cross_seats_sorted[1][0], "y": cross_seats_sorted[1][1]}
                    },
                    "column_five": {
                        "class_id": right_class_id_1,
                        "coordinates": {"x": right_seats_sorted[0][0], "y": right_seats_sorted[0][1]}
                    },
                    "column_six": {
                        "class_id": right_class_id_2,
                        "coordinates": {"x": right_seats_sorted[1][0], "y": right_seats_sorted[1][1]}
                    }
                }
                print(f"Last row: Left({left_class_id_1},{left_class_id_2}) Cross({cross_class_id_1},{cross_class_id_2}) Right({right_class_id_1},{right_class_id_2})")
            return row_data

        # Generate the row-based JSON output (with last row group)
        row_json_data = create_row_json_output_with_last_row(cross_aisle_pair_groups, aisle_pair, detections, last_row_group)
        print(f"✅ Generated seating layout with {len(row_json_data)} rows (including last row if present)")
        print("--- Seating Layout Output (row_json_data) ---")
        print(json.dumps(row_json_data, indent=2))
        print("--- End of Seating Layout Output ---")
        return row_json_data
    except Exception as e:
        print(f"❌ Error in process_seating_layout: {e}")
        return None


# Keep the original script functionality for command-line usage
if __name__ == "__main__":
    import os
    import sys
    
    # Load detection results
    webhook_file = os.getenv('WEBHOOK_DETECTION_FILE')
    use_webhook = "--webhook" in sys.argv

    if use_webhook and webhook_file and os.path.exists(webhook_file):
        print(f"Loading detection data from webhook file: {webhook_file}")
        try:
            with open(webhook_file, "r") as file:
                detections = json.load(file)
            print(f"Loaded {len(detections)} detections from webhook")
        except Exception as e:
            print(f"Error loading webhook file: {e}")
            detections = []
    else:
        # Default behavior - load from JSON_Data
        try:
            with open("JSON_Data/detection_results3.json", "r") as file:
                data = json.load(file)

            # ✅ Normalize to just a list of detections
            if isinstance(data, dict):
                if "detection_results" in data:
                    detections = data["detection_results"]
                elif "detections" in data:
                    detections = data["detections"]
                else:
                    detections = list(data.values())
            else:
                detections = data

            print(f"Loaded {len(detections)} detections from JSON_Data/detection_results3.json")

        except FileNotFoundError:
            print("Error: JSON_Data/detection_results3.json not found.")
            detections = []

    
    # Process the seating layout
    result = process_seating_layout(detections)
    
    if result:
        # Save the output
        with open("row_seating_layout.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Row-based seating layout saved to 'row_seating_layout.json'")
        print("--- Seating Layout Output (row_seating_layout.json) ---")
        print(json.dumps(result, indent=2))
        print("--- End of Seating Layout Output ---")
    else:
        print("Failed to process seating layout")


with open("JSON_Data/detection_results3.json", "r") as f:
    data = json.load(f)

print("DEBUG raw top-level keys:", data.keys() if isinstance(data, dict) else "not a dict")

if isinstance(data, dict):
    # check if any list inside
    for k, v in data.items():
        print(f"Key: {k}, Type: {type(v)}")



