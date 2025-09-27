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
    Create the final row-based JSON structure using actual detection coordinates
    """
    row_data = {}
    
    # Group all midpoints into rows by Y-coordinate proximity
    row_threshold = 60
    all_rows = []
    
    # Sort midpoints by Y coordinate (top to bottom)
    sorted_midpoints = sorted(midpoints, key=lambda pt: pt[1])
    
    for pt in sorted_midpoints:
        placed = False
        for row in all_rows:
            if abs(row[0][1] - pt[1]) < row_threshold:
                row.append(pt)
                placed = True
                break
        if not placed:
            all_rows.append([pt])
    
    # Sort each row by X coordinate (left to right)
    for row in all_rows:
        row.sort(key=lambda pt: pt[0])
    
    print(f"Found {len(all_rows)} rows of seats from detections")
    
    # Process each row - ONLY create seats for actual detections
    for row_idx, row_seats in enumerate(all_rows, 1):
        row_name = f"row_{row_idx}"
        row_data[row_name] = {}
        
        # Column names
        col_names = ["column_one", "column_two", "column_three", "column_four", "column_five", "column_six"]
        
        # Process ONLY the actual detected seats (no filling with dummy seats)
        for seat_idx, (seat_x, seat_y) in enumerate(row_seats):
            if seat_idx >= len(col_names):  # Safety check
                break
                
            col_name = col_names[seat_idx]
            
            # Find the class_id for this seat position
            class_id = find_class_id_by_coordinates(seat_x, seat_y, detections, tolerance=100)
            
            row_data[row_name][col_name] = {
                "class_id": class_id,
                "coordinates": {"x": seat_x, "y": seat_y}
            }
    
    return row_data


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


def find_class_id_by_coordinates(x: float, y: float, detections: List[Dict], tolerance: float = 50.0) -> int:
    """
    Find class_id for a seat at given coordinates by matching with detection data
    """
    for detection in detections:
        # Calculate detection center
        det_x = (detection["x_min"] + detection["x_max"]) / 2
        det_y = (detection["y_min"] + detection["y_max"]) / 2
        
        # Check if coordinates are close enough
        if abs(det_x - x) < tolerance and abs(det_y - y) < tolerance:
            return detection.get("class_id", 0)
    
    # Default to unoccupied if no match found
    return 1


def create_seat_data(seat_pos: Tuple[float, float], detection: Optional[Dict], row_idx: int, col_idx: int) -> Dict[str, Any]:
    """
    Create seat data dictionary for a single seat - matches seating.py format
    """
    if detection:
        class_id = detection.get("class_id", 0)
    else:
        class_id = 1  # Default unoccupied
    
    return {
        "class_id": class_id,
        "coordinates": {"x": float(seat_pos[0]), "y": float(seat_pos[1])}
    }
