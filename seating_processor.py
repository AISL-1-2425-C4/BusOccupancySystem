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
    try:
        detections = detections_input
        print(f"🚌 Processing {len(detections)} detections using full seating.py algorithm...")
        
        if not detections:
            return None
        
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
            """Find pairs within one side based on closest Y-coordinates"""
            if len(seats) < 2:
                return []
            
            if exclude_seats is None:
                exclude_seats = []
            
            # Filter out seats that are already in the cross-aisle pair
            available_seats = [seat for seat in seats if seat not in exclude_seats]
            
            if len(available_seats) < 2:
                return []
            
            # Sort seats from BOTTOM to TOP (highest Y to lowest Y)
            available_seats_sorted = sorted(available_seats, key=lambda pt: pt[1], reverse=True)
            
            side_pairs = []
            used = set()
            
            # For each unused seat, find its closest horizontal partner
            for i, seat1 in enumerate(available_seats_sorted):
                if i in used:
                    continue
                
                best_partner = None
                best_partner_idx = None
                min_y_distance = float('inf')
                
                # Find the closest seat by Y-coordinate (same row)
                for j, seat2 in enumerate(available_seats_sorted):
                    if i == j or j in used:
                        continue
                    
                    # Use pure Y-distance to find seats in the same row
                    y_distance = abs(seat1[1] - seat2[1])
                    
                    if y_distance < min_y_distance:
                        min_y_distance = y_distance
                        best_partner = seat2
                        best_partner_idx = j
                
                # If we found a partner, pair them
                if best_partner is not None:
                    side_pairs.append((seat1, best_partner))
                    used.add(i)
                    used.add(best_partner_idx)
            
            return side_pairs

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
            """Find the class_id of a detection by matching coordinates within tolerance"""
            for detection in detections:
                det_x_center = (detection["x_min"] + detection["x_max"]) / 2
                det_y_center = (detection["y_min"] + detection["y_max"]) / 2
                
                # Check if coordinates match within tolerance
                distance = ((det_x_center - x) ** 2 + (det_y_center - y) ** 2) ** 0.5
                if distance <= tolerance:
                    return detection["class_id"]
            
            # Default to unoccupied if no match found
            return 1

        # Create row-based JSON output
        def create_row_json_output(cross_aisle_pair_groups, aisle_pair, detections):
            """Create JSON output organized by rows with class_ids"""
            
            # Sort cross-aisle pair groups by Y-coordinate (bottom to top)
            sorted_groups = sorted(cross_aisle_pair_groups, 
                                  key=lambda group: group['y_center'], 
                                  reverse=True)
            
            row_data = {}
            
            # Process regular rows
            for row_idx, group in enumerate(sorted_groups, 1):
                row_name = f"row_{row_idx}"
                
                # Get left pair seats
                left_seat1 = group['left_pair'][0]
                left_seat2 = group['left_pair'][1]
                
                # Get right pair seats  
                right_seat1 = group['right_pair'][0]
                right_seat2 = group['right_pair'][1]
                
                # Sort left pair by X-coordinate (left to right)
                left_seats_sorted = sorted([left_seat1, left_seat2], key=lambda seat: seat[0])
                
                # Sort right pair by X-coordinate (left to right)
                right_seats_sorted = sorted([right_seat1, right_seat2], key=lambda seat: seat[0])
                
                # Find class_ids for each seat
                left_class_id_1 = find_class_id_by_coordinates(left_seats_sorted[0][0], left_seats_sorted[0][1], detections)
                left_class_id_2 = find_class_id_by_coordinates(left_seats_sorted[1][0], left_seats_sorted[1][1], detections)
                right_class_id_1 = find_class_id_by_coordinates(right_seats_sorted[0][0], right_seats_sorted[0][1], detections)
                right_class_id_2 = find_class_id_by_coordinates(right_seats_sorted[1][0], right_seats_sorted[1][1], detections)
                
                # Create row object
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
                
                print(f"Row {row_idx}: Left({left_class_id_1},{left_class_id_2}) Right({right_class_id_1},{right_class_id_2})")
            
            return row_data

        # Generate the row-based JSON output
        row_json_data = create_row_json_output(cross_aisle_pair_groups, aisle_pair, detections)
        
        print(f"✅ Generated seating layout with {len(row_json_data)} rows")
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
                detections = json.load(file)
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
    else:
        print("Failed to process seating layout")
