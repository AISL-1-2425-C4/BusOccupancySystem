"""
Seating processing library - extracted from seating.py
Contains the core detection processing logic as reusable functions
"""

# import numpy as np  # Removed for Vercel compatibility
import json
from typing import List, Dict, Any, Tuple, Optional

def process_detections_to_layout(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Process detection results into seating layout using the full seating.py algorithm
    
    Args:
        detections: List of detection dictionaries with x_min, y_min, x_max, y_max, class_id, etc.
    
    Returns:
        Dictionary with row-based seating layout or None if processing fails
    """
    try:
        if not detections:
            return None
        
        print(f"🚀 Processing {len(detections)} detections...")
        
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
        row_threshold = 60  # pixels
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

        # Find optimal aisle position using the seating.py algorithm
        x_coords = [pt[0] for pt in midpoints]
        x_min_coord, x_max_coord = min(x_coords), max(x_coords)
        total_seats = len(x_coords)
        
        print(f"Total seats: {total_seats}")
        
        # Find best aisle position for balanced split
        best_balance = float('inf')
        best_aisle_pos = x_min_coord + (x_max_coord - x_min_coord) / 2
        
        # Test aisle positions across the width (pure Python implementation)
        width_range = x_max_coord - x_min_coord
        start_pos = x_min_coord + width_range * 0.1
        end_pos = x_min_coord + width_range * 0.9
        step = (end_pos - start_pos) / 199  # 200 points = 199 intervals
        test_positions = [start_pos + i * step for i in range(200)]
        
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
        
        # Separate seats into left and right groups
        left_seats = [pt for pt in midpoints if pt[0] < aisle_position]
        right_seats = [pt for pt in midpoints if pt[0] >= aisle_position]
        
        print(f"Left side seats: {len(left_seats)}, Right side seats: {len(right_seats)}")
        
        # Organize seats by rows for each side
        left_rows = organize_seats_by_rows(left_seats)
        right_rows = organize_seats_by_rows(right_seats)
        
        # Create the final row-based JSON structure
        row_json_data = create_row_json_structure(left_rows, right_rows, detections, midpoints)
        
        print(f"Generated seating layout with {len(row_json_data)} rows")
        return row_json_data
        
    except Exception as e:
        print(f"❌ Error in process_detections_to_layout: {e}")
        return None


def organize_seats_by_rows(seats: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Organize seats into rows by y-coordinate proximity
    """
    if not seats:
        return {}
    
    # Sort by y-coordinate (top to bottom)
    seats_sorted = sorted(seats, key=lambda pt: pt[1])
    
    # Group into rows by y proximity
    rows = []
    row_threshold = 60
    for pt in seats_sorted:
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
    
    # Convert to dictionary format
    row_dict = {}
    for i, row in enumerate(rows):
        row_dict[f"row_{i+1}"] = row
    
    return row_dict


def create_row_json_structure(left_rows: Dict, right_rows: Dict, detections: List[Dict], midpoints: List[Tuple]) -> Dict[str, Any]:
    """
    Create the final row-based JSON structure matching the expected format
    """
    row_json_data = {}
    
    # Determine the maximum number of rows
    max_left_rows = len(left_rows)
    max_right_rows = len(right_rows)
    max_rows = max(max_left_rows, max_right_rows)
    
    # Standard bus layout: 10 rows, with row 10 having 6 seats
    target_rows = 10
    
    for row_idx in range(1, target_rows + 1):
        row_name = f"row_{row_idx}"
        row_json_data[row_name] = {}
        
        # Determine number of columns for this row
        if row_idx == 10:  # Last row has 6 seats
            columns = ["column_one", "column_two", "column_three", "column_four", "column_five", "column_six"]
        else:  # Other rows have 4 seats
            columns = ["column_one", "column_two", "column_three", "column_four"]
        
        # Get seats for this row from left and right sides
        left_row_key = f"row_{row_idx}"
        right_row_key = f"row_{row_idx}"
        
        left_seats = left_rows.get(left_row_key, [])
        right_seats = right_rows.get(right_row_key, [])
        
        # Assign seats to columns
        all_row_seats = []
        
        # Add left side seats (columns 1-2)
        for i, seat_pos in enumerate(left_seats[:2]):
            if i < len(columns):
                detection = find_matching_detection(seat_pos, detections, midpoints)
                row_json_data[row_name][columns[i]] = create_seat_data(seat_pos, detection, row_idx, i+1)
                all_row_seats.append(seat_pos)
        
        # Add right side seats (columns 3-4 or 3-6 for last row)
        right_start_col = 2
        for i, seat_pos in enumerate(right_seats):
            col_idx = right_start_col + i
            if col_idx < len(columns):
                detection = find_matching_detection(seat_pos, detections, midpoints)
                row_json_data[row_name][columns[col_idx]] = create_seat_data(seat_pos, detection, row_idx, col_idx+1)
                all_row_seats.append(seat_pos)
        
        # Fill remaining columns with default unoccupied seats if needed
        for i, col_name in enumerate(columns):
            if col_name not in row_json_data[row_name]:
                # Create default seat position
                default_x = 30 + (row_idx - 1) * 55
                if i < 2:  # Left side
                    default_y = 25 + i * 55
                else:  # Right side
                    default_y = 245 + (i - 2) * 55
                
                row_json_data[row_name][col_name] = {
                    "class_id": 1,
                    "class_name": "unoccupied",
                    "confidence": 0.5,
                    "position": {"x": float(default_x), "y": float(default_y)}
                }
    
    return row_json_data


def find_matching_detection(seat_pos: Tuple[float, float], detections: List[Dict], midpoints: List[Tuple]) -> Optional[Dict]:
    """
    Find the detection that corresponds to a seat position
    """
    seat_x, seat_y = seat_pos
    
    # Find the detection with midpoint closest to this seat position
    for i, (mid_x, mid_y) in enumerate(midpoints):
        if abs(mid_x - seat_x) < 50 and abs(mid_y - seat_y) < 50:  # Tolerance
            if i < len(detections):
                return detections[i]
    
    return None


def create_seat_data(seat_pos: Tuple[float, float], detection: Optional[Dict], row_idx: int, col_idx: int) -> Dict[str, Any]:
    """
    Create seat data dictionary for a single seat
    """
    # Calculate display position (scaled for frontend)
    display_x = 30 + (row_idx - 1) * 55
    if col_idx <= 2:  # Left side
        display_y = 25 + (col_idx - 1) * 55
    else:  # Right side
        display_y = 245 + (col_idx - 3) * 55
    
    if detection:
        # Use detection data
        class_id = detection.get("class_id", 0)
        class_name = detection.get("class_name", "occupied").lower()
        confidence = detection.get("confidence", 0.5)
        
        # Normalize class_id: 0 = occupied, 1 = unoccupied
        if class_id == 0 or "occupied" in class_name:
            final_class_id = 0
            final_class_name = "occupied"
        else:
            final_class_id = 1
            final_class_name = "unoccupied"
    else:
        # Default unoccupied
        final_class_id = 1
        final_class_name = "unoccupied"
        confidence = 0.5
    
    return {
        "class_id": final_class_id,
        "class_name": final_class_name,
        "confidence": confidence,
        "position": {"x": float(display_x), "y": float(display_y)}
    }
