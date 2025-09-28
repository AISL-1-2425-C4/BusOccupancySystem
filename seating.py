import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Load detection results
webhook_file = os.getenv('WEBHOOK_DETECTION_FILE')
use_webhook = "--webhook" in sys.argv

def load_detection_data(file_path):
    """Load detection data from JSON file, handling both old and new formats"""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # Handle new format with 'detections' key
        if isinstance(data, dict) and 'detections' in data:
            detections = data['detections']
            print(f"Loaded {len(detections)} detections from new format (with 'detections' key)")
            if 'inference_time_sec' in data:
                print(f"Inference time: {data['inference_time_sec']} seconds")
        # Handle old format (direct array)
        elif isinstance(data, list):
            detections = data
            print(f"Loaded {len(detections)} detections from old format (direct array)")
        # Handle old format with 'detection_results' key
        elif isinstance(data, dict) and 'detection_results' in data:
            detections = data['detection_results']
            print(f"Loaded {len(detections)} detections from old format (with 'detection_results' key)")
        else:
            print("Error: Unrecognized JSON format")
            detections = []
            
        return detections
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

if use_webhook and webhook_file and os.path.exists(webhook_file):
    print(f"Loading detection data from webhook file: {webhook_file}")
    detections = load_detection_data(webhook_file)
else:
    # Default behavior - load from JSON_Data
    detections = load_detection_data("JSON_Data/detection_results3.json")

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

# Divide canvas into four columns
canvas_width, canvas_height = 4608, 2592
col_edges = [0, canvas_width/4, canvas_width/2, 3*canvas_width/4, canvas_width]

def assign_to_column(x):
    for i in range(4):
        if col_edges[i] <= x < col_edges[i+1]:
            return i
    return 3  # rightmost column


# Organize each row to have one seat per column (except last row)


# Improved row grouping: always group closest y values into rows
midpoints_sorted = sorted(midpoints, key=lambda pt: pt[1])
rows = []
row_threshold = 60
for pt in midpoints_sorted:
    placed = False
    for row in rows:
        if abs(row[0][1] - pt[1]) < row_threshold:
            row.append(pt)
            placed = True
            break
    if not placed:
        rows.append([pt])

# For each row, find the closest x partner for each seat
partner_pairs = []
for row in rows:
    used = set()
    row_sorted = sorted(row, key=lambda pt: pt[0])
    for i, pt in enumerate(row_sorted):
        if i in used:
            continue
        # Find closest x partner not already used
        min_dist = float('inf')
        partner_idx = None
        for j, other in enumerate(row_sorted):
            if j == i or j in used:
                continue
            dist = abs(pt[0] - other[0])
            if dist < min_dist:
                min_dist = dist
                partner_idx = j
        if partner_idx is not None:
            partner_pairs.append((pt, row_sorted[partner_idx]))
            used.add(i)
            used.add(partner_idx)
    # If odd number, last seat is unpaired
    if len(row_sorted) % 2 == 1:
        for k in range(len(row_sorted)):
            if k not in used:
                partner_pairs.append((row_sorted[k], None))
                break

# Improved aisle detection - iteratively adjust until balanced (disparity ≤ 1)

x_coords = [pt[0] for pt in midpoints]
x_min, x_max = min(x_coords), max(x_coords)
total_seats = len(x_coords)
target_left = total_seats // 2  # Target roughly half the seats on each side

print(f"Total seats: {total_seats}, Target per side: ~{target_left}")

# Start with a more comprehensive search for the optimal aisle position
best_balance = float('inf')
best_aisle_pos = x_min + (x_max - x_min) / 2  # default to middle

# Test aisle positions across the entire width with fine granularity
test_positions = np.linspace(x_min + (x_max - x_min) * 0.1, 
                            x_min + (x_max - x_min) * 0.9, 200)

for test_pos in test_positions:
    left_count = sum(1 for x in x_coords if x < test_pos)
    right_count = sum(1 for x in x_coords if x >= test_pos)
    
    # Calculate how unbalanced this split is
    balance_score = abs(left_count - right_count)
    
    # Prioritize positions that create the most balanced split
    if balance_score < best_balance:
        best_balance = balance_score
        best_aisle_pos = test_pos
        
        # If we found a perfect or near-perfect balance, we can stop
        if balance_score <= 1:
            print(f"Found excellent balance at x = {test_pos:.1f} (disparity: {balance_score})")
            break

aisle_position = best_aisle_pos

# Verify the final balance
left_count = sum(1 for x in x_coords if x < aisle_position)
right_count = sum(1 for x in x_coords if x >= aisle_position)
disparity = abs(left_count - right_count)

print(f"Final aisle position: x = {aisle_position:.1f}")
print(f"Final balance - Left: {left_count}, Right: {right_count}")
print(f"Disparity: {disparity} (Target: ≤ 1)")

if disparity <= 1:
    print("✓ Achieved balanced split!")
else:
    print(f"⚠ Still unbalanced by {disparity} seats")

# Separate seats into left and right groups based on aisle position
left_seats = [pt for pt in midpoints if pt[0] < aisle_position]
right_seats = [pt for pt in midpoints if pt[0] >= aisle_position]

print(f"Left side seats: {len(left_seats)}")
print(f"Right side seats: {len(right_seats)}")
print(f"Total seats in left+right: {len(left_seats) + len(right_seats)}")
print(f"Expected total seats: 41")
print(f"X coordinate range: {x_min:.1f} to {x_max:.1f}")
print(f"Aisle position: {aisle_position:.1f} ({((aisle_position - x_min) / (x_max - x_min) * 100):.1f}% from left)")

# Verify we have all seats
if len(left_seats) + len(right_seats) != len(midpoints):
    print("WARNING: Some seats may be missing from left/right classification!")
else:
    print("✓ All seats successfully classified into left/right groups")

# Find the ONE cross-aisle pair with closest x-values to the aisle
aisle_pair = None
min_x_distance_sum = float('inf')

# Find the pair of seats (one from each side) that are closest to the aisle in x-coordinates
for left_pt in left_seats:
    for right_pt in right_seats:
        # Calculate how close both seats are to the aisle
        left_distance_to_aisle = abs(left_pt[0] - aisle_position)
        right_distance_to_aisle = abs(right_pt[0] - aisle_position)
        total_distance_to_aisle = left_distance_to_aisle + right_distance_to_aisle
        
        if total_distance_to_aisle < min_x_distance_sum:
            min_x_distance_sum = total_distance_to_aisle
            aisle_pair = (left_pt, right_pt)

print(f"Cross-aisle pair found: {aisle_pair is not None}")
if aisle_pair:
    left_dist = abs(aisle_pair[0][0] - aisle_position)
    right_dist = abs(aisle_pair[1][0] - aisle_position)
    print(f"  Single cross-aisle pair: Left({aisle_pair[0][0]:.1f}, {aisle_pair[0][1]:.1f}) <-> Right({aisle_pair[1][0]:.1f}, {aisle_pair[1][1]:.1f})")
    print(f"  Left seat distance to aisle: {left_dist:.1f}")
    print(f"  Right seat distance to aisle: {right_dist:.1f}")
else:
    print("  No cross-aisle pair found!")

# Find pairs within each side based on closest Y-coordinates (excluding cross-aisle pair nodes)
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
    print(f"Excluding cross-aisle seats from side pairing: {len(cross_aisle_seats)} seats")

left_side_pairs = find_side_pairs(left_seats, "Left", exclude_seats=[aisle_pair[0]] if aisle_pair else [])
right_side_pairs = find_side_pairs(right_seats, "Right", exclude_seats=[aisle_pair[1]] if aisle_pair else [])

# Now create pairs of pairs (group pairs together based on Y-difference)
def pair_pairs_by_y(pairs, side_name):
    """Pair existing pairs together based on Y-coordinate proximity"""
    if len(pairs) < 2:
        return []
    
    # Calculate center Y-coordinate for each pair
    pair_centers = []
    for pair in pairs:
        center_y = (pair[0][1] + pair[1][1]) / 2
        pair_centers.append((pair, center_y))
    
    # Sort pairs by their Y-coordinate (bottom to top)
    pair_centers_sorted = sorted(pair_centers, key=lambda x: x[1], reverse=True)
    
    pair_of_pairs = []
    used = set()
    
    print(f"\n{side_name} side - Pairing pairs together:")
    
    # For each unused pair, find its closest Y-coordinate partner pair
    for i, (pair1, y1) in enumerate(pair_centers_sorted):
        if i in used:
            continue
        
        best_partner_pair = None
        best_partner_idx = None
        min_y_distance = float('inf')
        
        # Find the closest pair by Y-coordinate
        for j, (pair2, y2) in enumerate(pair_centers_sorted):
            if i == j or j in used:
                continue
            
            y_distance = abs(y1 - y2)
            
            if y_distance < min_y_distance:
                min_y_distance = y_distance
                best_partner_pair = pair2
                best_partner_idx = j
        
        # If we found a partner pair, group them
        if best_partner_pair is not None:
            pair_of_pairs.append((pair1, best_partner_pair))
            used.add(i)
            used.add(best_partner_idx)
            print(f"  Grouped pairs: Y-diff={min_y_distance:.1f}")
        else:
            # If no partner found, this pair will be handled specially
            print(f"  Unpaired pair found (will be handled specially)")
    
    print(f"{side_name} side: {len(pair_of_pairs)} pair-of-pairs created")
    return pair_of_pairs

# Pair LEFT pairs with RIGHT pairs (cross-aisle pair pairing) - bottom to top, one-to-one
def pair_left_with_right_pairs(left_pairs, right_pairs):
    """Pair left side pairs with right side pairs based on Y-coordinate proximity, bottom to top"""
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
    
    # Sort pairs by their Y-coordinate (bottom to top - highest Y first)
    left_pair_centers_sorted = sorted(left_pair_centers, key=lambda x: x[1], reverse=True)
    right_pair_centers_sorted = sorted(right_pair_centers, key=lambda x: x[1], reverse=True)
    
    cross_aisle_pair_groups = []
    used_left = set()
    used_right = set()
    
    print(f"\nPairing LEFT pairs with RIGHT pairs (bottom to top, one-to-one):")
    print(f"Left pairs available: {len(left_pairs)}, Right pairs available: {len(right_pairs)}")
    
    # Debug: Show all pairs with their Y-coordinates
    print("Left pairs (sorted bottom to top):")
    for i, (pair, y) in enumerate(left_pair_centers_sorted):
        print(f"  {i}: Y={y:.1f}, seats=({pair[0][0]:.1f},{pair[0][1]:.1f})-({pair[1][0]:.1f},{pair[1][1]:.1f})")
    
    print("Right pairs (sorted bottom to top):")
    for i, (pair, y) in enumerate(right_pair_centers_sorted):
        print(f"  {i}: Y={y:.1f}, seats=({pair[0][0]:.1f},{pair[0][1]:.1f})-({pair[1][0]:.1f},{pair[1][1]:.1f})")
    
    # Process all pairs to ensure everyone gets paired
    max_pairs = max(len(left_pairs), len(right_pairs))
    
    # Simple approach: pair in order from bottom to top
    min_pairs = min(len(left_pairs), len(right_pairs))
    
    for i in range(min_pairs):
        left_pair, left_y = left_pair_centers_sorted[i]
        right_pair, right_y = right_pair_centers_sorted[i]
        
        y_diff = abs(left_y - right_y)
        cross_aisle_pair_groups.append({
            'left_pair': left_pair,
            'right_pair': right_pair,
            'y_difference': y_diff,
            'total_chairs': 4  # 2 left + 2 right = 4 chairs
        })
        
        print(f"  Pair {i + 1}: Left(Y={left_y:.1f}) <-> Right(Y={right_y:.1f}), Y-diff={y_diff:.1f}")
    
    # Report any remaining unpaired pairs
    if len(left_pairs) > len(right_pairs):
        print(f"  {len(left_pairs) - len(right_pairs)} left pairs remain unpaired")
    elif len(right_pairs) > len(left_pairs):
        print(f"  {len(right_pairs) - len(left_pairs)} right pairs remain unpaired")
    
    print(f"Cross-aisle pair groups created: {len(cross_aisle_pair_groups)} (4 chairs each)")
    
    return cross_aisle_pair_groups

# Restore cross-aisle pair group logic: pair left and right pairs by y-proximity as normal
cross_aisle_pair_groups = pair_left_with_right_pairs(left_side_pairs, right_side_pairs)

# Debug: Show what cross-aisle pair groups were created
print(f"\nDEBUG: Cross-aisle pair groups created:")
for i, group in enumerate(cross_aisle_pair_groups):
    left_pair = group['left_pair']
    right_pair = group['right_pair']
    left_y = (left_pair[0][1] + left_pair[1][1]) / 2
    right_y = (right_pair[0][1] + right_pair[1][1]) / 2
    print(f"  Group {i+1}: Left Y={left_y:.1f} <-> Right Y={right_y:.1f}")
    print(f"    Left seats: ({left_pair[0][0]:.1f},{left_pair[0][1]:.1f})-({left_pair[1][0]:.1f},{left_pair[1][1]:.1f})")
    print(f"    Right seats: ({right_pair[0][0]:.1f},{right_pair[0][1]:.1f})-({right_pair[1][0]:.1f},{right_pair[1][1]:.1f})")

# Handle the last row grouping - find the bottom-most pairs for special visualization
print(f"\nCreating last row grouping visualization:")
last_row_group = None
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
        print(f"Last row group created with 6 chairs (with cross-aisle pair):")
    else:
        left_y = (left_lowest_pair[0][1] + left_lowest_pair[1][1]) / 2
        right_y = (right_lowest_pair[0][1] + right_lowest_pair[1][1]) / 2
        if left_y > right_y:
            # Left is lower, generate a virtual pair on the right for left_lowest_pair
            cross_aisle = generate_virtual_pair_for(left_lowest_pair, 'left')
            last_row_group = {
                'bottom_left_pair': left_lowest_pair,
                'cross_aisle_pair': cross_aisle,
                'bottom_right_pair': right_lowest_pair,
                'total_chairs': 6
            }
            print(f"Last row group created with 6 chairs (left pair with virtual cross-aisle pair):")
        else:
            # Right is lower, generate a virtual pair on the left for right_lowest_pair
            cross_aisle = generate_virtual_pair_for(right_lowest_pair, 'right')
            last_row_group = {
                'bottom_left_pair': left_lowest_pair,
                'cross_aisle_pair': cross_aisle,
                'bottom_right_pair': right_lowest_pair,
                'total_chairs': 6
            }
            print(f"Last row group created with 6 chairs (right pair with virtual cross-aisle pair):")
    print(f"  Bottom left pair: 2 chairs")
    print(f"  Cross-aisle pair: 2 chairs")
    print(f"  Bottom right pair: 2 chairs")

    # Remove any cross-aisle pair group that matches the last row's left and right pairs
    to_remove = []
    for group in cross_aisle_pair_groups:
        if (group['left_pair'] == left_lowest_pair and group['right_pair'] == right_lowest_pair):
            to_remove.append(group)
    for group in to_remove:
        cross_aisle_pair_groups.remove(group)
else:
    print("Not enough data to create last row group.")
    last_row_group = None

# Plot seats with all pairings and groupings
plt.figure(figsize=(canvas_width/400, canvas_height/400))
plt.xlim(0, canvas_width)
plt.ylim(canvas_height, 0)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Bus Seats: Pairs + Pair-of-Pairs + Last Row (6 chairs)")
plt.xlabel("X")
plt.ylabel("Y")

# Plot left side seats in blue
for pt in left_seats:
    plt.plot(pt[0], pt[1], 'bo', markersize=8, label='Left Side' if pt == left_seats[0] else "")

# Plot right side seats in red
for pt in right_seats:
    plt.plot(pt[0], pt[1], 'ro', markersize=8, label='Right Side' if pt == right_seats[0] else "")

# Draw aisle line
plt.axvline(x=aisle_position, color='green', linestyle='--', linewidth=2, label='Aisle')

# Draw the cross-aisle pairing line (part of last row)
if aisle_pair:
    left_pt, right_pt = aisle_pair
    print(f"Drawing cross-aisle line (last row) from ({left_pt[0]:.1f}, {left_pt[1]:.1f}) to ({right_pt[0]:.1f}, {right_pt[1]:.1f})")
    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], 
             'purple', linewidth=4, alpha=0.9, label='Cross-Aisle (Last Row)')

# Draw individual pair lines (thin lines)
for pt1, pt2 in left_side_pairs:
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
             'cyan', linewidth=1, alpha=0.5)

for pt1, pt2 in right_side_pairs:
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
             'orange', linewidth=1, alpha=0.5)

# Draw cross-aisle pair group connections (thick lines connecting left pairs to right pairs)
colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'darkmagenta', 'darkcyan']
color_idx = 0

print(f"\nDEBUG: Drawing {len(cross_aisle_pair_groups)} cross-aisle pair group lines:")

for group in cross_aisle_pair_groups:
    left_pair = group['left_pair']
    right_pair = group['right_pair']
    
    # Calculate center of left pair
    left_center_x = (left_pair[0][0] + left_pair[1][0]) / 2
    left_center_y = (left_pair[0][1] + left_pair[1][1]) / 2
    
    # Calculate center of right pair
    right_center_x = (right_pair[0][0] + right_pair[1][0]) / 2
    right_center_y = (right_pair[0][1] + right_pair[1][1]) / 2
    
    print(f"  Drawing line from ({left_center_x:.1f},{left_center_y:.1f}) to ({right_center_x:.1f},{right_center_y:.1f})")
    
    # Draw connection between left pair center and right pair center
    plt.plot([left_center_x, right_center_x], [left_center_y, right_center_y], 
             colors[color_idx % len(colors)], linewidth=3, alpha=0.8)
    color_idx += 1

if len(cross_aisle_pair_groups) == 0:
    print("  WARNING: No cross-aisle pair groups to draw!")

# Draw special last row connection (blue pair -> cross-aisle pair -> red pair)
if last_row_group:
    print(f"\nDrawing last row connection (6-chair group):")
    
    # Calculate centers
    left_pair = last_row_group['bottom_left_pair']
    cross_pair = last_row_group['cross_aisle_pair']
    right_pair = last_row_group['bottom_right_pair']
    
    left_center_x = (left_pair[0][0] + left_pair[1][0]) / 2
    left_center_y = (left_pair[0][1] + left_pair[1][1]) / 2
    
    cross_center_x = (cross_pair[0][0] + cross_pair[1][0]) / 2
    cross_center_y = (cross_pair[0][1] + cross_pair[1][1]) / 2
    
    right_center_x = (right_pair[0][0] + right_pair[1][0]) / 2
    right_center_y = (right_pair[0][1] + right_pair[1][1]) / 2
    
    print(f"  Left pair center: ({left_center_x:.1f}, {left_center_y:.1f})")
    print(f"  Cross-aisle center: ({cross_center_x:.1f}, {cross_center_y:.1f})")
    print(f"  Right pair center: ({right_center_x:.1f}, {right_center_y:.1f})")
    
    # Draw connection lines for last row grouping
    # Left pair -> Cross-aisle pair
    plt.plot([left_center_x, cross_center_x], [left_center_y, cross_center_y], 
             'gold', linewidth=4, alpha=0.9, linestyle='-')
    
    # Cross-aisle pair -> Right pair  
    plt.plot([cross_center_x, right_center_x], [cross_center_y, right_center_y], 
             'gold', linewidth=4, alpha=0.9, linestyle='-')
    
    print(f"  Drew last row connection lines in gold")

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Left Side'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Right Side'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Aisle'),
    Line2D([0], [0], color='purple', linewidth=4, label='Cross-Aisle Pair'),
    Line2D([0], [0], color='cyan', linewidth=1, label='Individual Pairs (Same Side)'),
    Line2D([0], [0], color='darkblue', linewidth=3, label='Cross-Aisle Pair Groups (Blue↔Red)'),
    Line2D([0], [0], color='gold', linewidth=4, label='Last Row Connection (6 chairs)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.savefig("midpoints_columns.png")
plt.show()




# Use ALL seats for left/right separation (including the back middle seats)
print(f"Total seats detected: {len(midpoints)}")

# Organize all seats into rows for the row-based visualization
all_seats_sorted = sorted(midpoints, key=lambda pt: pt[1])
num_rows = 5  # Adjust based on your bus layout
seats_per_row = len(all_seats_sorted) // num_rows
extra = len(all_seats_sorted) % num_rows
rows = []
start = 0
for i in range(num_rows):
    end = start + seats_per_row + (1 if i < extra else 0)
    rows.append(all_seats_sorted[start:end])
    start = end

# --- Separate seat midpoints into four columns with equal number of dots ---


# For each column, separate into rows of two seats each
    # Removed column plotting code

# Organize each row to have one seat per column (except last row)
final_rows = []
seat_map = {}


extra_seats = []
for idx, row in enumerate(rows):
    row_sorted = sorted(row, key=lambda pt: pt[0])
    row_seats = row_sorted[:4]
    final_rows.append(row_seats)
    if len(row_sorted) > 4:
        extra_seats.append(row_sorted[4:])

# Add extra seats as new rows
for extra in extra_seats:
    final_rows.append(extra)


# Prepare JSON mapping with left/right grouping (ALL 41 seats)
seat_map = {
    "left_side": left_seats,
    "right_side": right_seats,
    "aisle_position": aisle_position,
    "total_seats": len(midpoints),
    "left_count": len(left_seats),
    "right_count": len(right_seats)
}

# Also organize by rows for each side
def organize_seats_by_rows(seats, side_name):
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
    
    # Add to seat map
    row_map = {}
    for i, row in enumerate(rows):
        row_map[f"{side_name}_row_{i+1}"] = row
    
    return row_map

# Add organized rows to seat map
seat_map.update(organize_seats_by_rows(left_seats, "left"))
seat_map.update(organize_seats_by_rows(right_seats, "right"))

# Create function to find class_id from coordinates
def find_class_id_by_coordinates(x, y, detections, tolerance=50):
    """Find the class_id of a detection by matching coordinates within tolerance"""
    """Find the class_id of a detection by matching coordinates within tolerance, or return 3 for generated dots."""
    # Check if this is a generated virtual dot
    if hasattr(find_side_pairs, 'generated_virtual_dots'):
        for vdot in find_side_pairs.generated_virtual_dots:
            if abs(vdot[0] - x) < 1e-3 and abs(vdot[1] - y) < 1e-3:
                return 2  # class_id 2 for unknown/generated
    for detection in detections:
        det_x_center = (detection["x_min"] + detection["x_max"]) / 2
        det_y_center = (detection["y_min"] + detection["y_max"]) / 2
        
        # Check if coordinates match within tolerance
        if abs(det_x_center - x) < tolerance and abs(det_y_center - y) < tolerance:
            return detection["class_id"]
    return None  # Not found

# Create row-based JSON output with class_ids
def create_row_json_output(cross_aisle_pair_groups, aisle_pair, detections, last_row_group):
    """Create JSON output organized by rows (bottom to top) with class_ids"""
    
    # Sort cross-aisle pair groups by Y-coordinate (bottom to top - highest Y first)
    sorted_groups = sorted(cross_aisle_pair_groups, 
                          key=lambda group: ((group['left_pair'][0][1] + group['left_pair'][1][1]) / 2 + 
                                           (group['right_pair'][0][1] + group['right_pair'][1][1]) / 2) / 2, 
                          reverse=True)
    
    row_data = {}
    
    # Find which group is part of the last row (connected with gold lines)
    last_row_left_pair = None
    last_row_right_pair = None
    if last_row_group:
        last_row_left_pair = last_row_group['bottom_left_pair']
        last_row_right_pair = last_row_group['bottom_right_pair']
    
    # Process regular rows (excluding the one that's part of the last row)
    regular_row_idx = 1
    for group in sorted_groups:
        # Skip if this group is part of the last row
        if (last_row_left_pair and last_row_right_pair and 
            group['left_pair'] == last_row_left_pair and group['right_pair'] == last_row_right_pair):
            continue
            
        row_name = f"row_{regular_row_idx}"
        
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
        
        print(f"Row {regular_row_idx}: Left({left_class_id_1},{left_class_id_2}) Right({right_class_id_1},{right_class_id_2})")
        regular_row_idx += 1
    
    # Create the special LAST ROW with 6 seats (connected by gold lines)
    if last_row_group and aisle_pair:
        # Get the three pairs that make up the last row
        left_pair = last_row_group['bottom_left_pair']
        cross_pair = last_row_group['cross_aisle_pair'] 
        right_pair = last_row_group['bottom_right_pair']
        
        # Sort each pair by X-coordinate
        left_seats_sorted = sorted([left_pair[0], left_pair[1]], key=lambda seat: seat[0])
        right_seats_sorted = sorted([right_pair[0], right_pair[1]], key=lambda seat: seat[0])
        cross_seats_sorted = sorted([cross_pair[0], cross_pair[1]], key=lambda seat: seat[0])
        
        # Find class_ids for all 6 seats
        left_class_id_1 = find_class_id_by_coordinates(left_seats_sorted[0][0], left_seats_sorted[0][1], detections)
        left_class_id_2 = find_class_id_by_coordinates(left_seats_sorted[1][0], left_seats_sorted[1][1], detections)
        cross_class_id_1 = find_class_id_by_coordinates(cross_seats_sorted[0][0], cross_seats_sorted[0][1], detections)
        cross_class_id_2 = find_class_id_by_coordinates(cross_seats_sorted[1][0], cross_seats_sorted[1][1], detections)
        right_class_id_1 = find_class_id_by_coordinates(right_seats_sorted[0][0], right_seats_sorted[0][1], detections)
        right_class_id_2 = find_class_id_by_coordinates(right_seats_sorted[1][0], right_seats_sorted[1][1], detections)
        
        # Create the last row with 6 columns
        row_data[f"row_{regular_row_idx}"] = {
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
        
        print(f"LAST ROW {regular_row_idx} (6 seats): Left({left_class_id_1},{left_class_id_2}) Cross({cross_class_id_1},{cross_class_id_2}) Right({right_class_id_1},{right_class_id_2})")
    
    return row_data

# Generate the row-based JSON output
print(f"\nGenerating row-based JSON output:")
row_json_data = create_row_json_output(cross_aisle_pair_groups, aisle_pair, detections, last_row_group)

# Save row-based JSON
with open("row_seating_layout.json", "w") as f:
    json.dump(row_json_data, f, indent=2)

print(f"Row-based seating layout saved to 'row_seating_layout.json'")

# Also save the original seat mapping
with open("seat_mapping.json", "w") as f:
    json.dump(seat_map, f, indent=2)

# Extra visualizations removed - keeping only the main seat pairing visualization
