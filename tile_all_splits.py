import os
import cv2

def tile_image_left_right_with_overlap(img_path, label_path, out_img_dir, out_lbl_dir, overlap_pct=0.1):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Couldn't load image: {img_path}")
        return

    h, w, _ = image.shape
    tile_w = w // 2
    overlap_w = int(tile_w * overlap_pct)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Read YOLO labels (class x_center y_center width height)
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls, xc, yc, bw, bh = map(float, parts[:5])
                        abs_xc = xc * w
                        abs_yc = yc * h
                        abs_bw = bw * w
                        abs_bh = bh * h
                        labels.append((cls, abs_xc, abs_yc, abs_bw, abs_bh))
                    except:
                        continue

    tile_id = 0
    # Left Tile (0 to 2304)
    x0_left = 0
    x1_left = tile_w + 2 * overlap_w  # Add overlap to the right edge
    # Right Tile (2304 to 4608)
    x0_right = tile_w - overlap_w  # Add overlap to the left edge
    x1_right = w

    # Clamp to image boundaries
    x0_left = max(0, x0_left)
    x1_left = min(w, x1_left)
    x0_right = max(0, x0_right)
    x1_right = min(w, x1_right)

    # Create tiles
    tiles = [
        (x0_left, x1_left, "left"),
        (x0_right, x1_right, "right")
    ]

    for x0, x1, side in tiles:
        # Clip the image to the tile
        tile = image[:, x0:x1]
        tile_img_name = f"{base_name}_tile_{side}.jpg"
        tile_lbl_name = tile_img_name.replace(".jpg", ".txt")

        # Save tile image
        cv2.imwrite(os.path.join(out_img_dir, tile_img_name), tile)

        # Adjust and filter labels for this tile
        tile_labels = []
        tile_w_actual = x1 - x0

        for cls, abs_xc, abs_yc, abs_bw, abs_bh in labels:
            if (x0 <= abs_xc <= x1):
                rel_x = (abs_xc - x0) / tile_w_actual
                rel_w = abs_bw / tile_w_actual

                # Clamp to valid YOLO range
                rel_x = max(0.0, min(1.0, rel_x))
                rel_w = max(0.0, min(1.0, rel_w))

                tile_labels.append(f"{int(cls)} {rel_x:.6f} {abs_yc / h:.6f} {rel_w:.6f} {abs_bh / h:.6f}")

        # Save label file
        with open(os.path.join(out_lbl_dir, tile_lbl_name), "w") as f:
            f.write("\n".join(tile_labels))

        tile_id += 1


def tile_split_left_right(split, input_base="path/dataset", output_base="path/dataset", overlap_pct=0.01):
    print(f"Tiling '{split}' set into 2 parts (left-right) with {int(overlap_pct*100)}% overlap...")

    in_img_dir = os.path.join(input_base, split, "images")
    in_lbl_dir = os.path.join(input_base, split, "labels")
    out_img_dir = os.path.join(output_base, split, "images")
    out_lbl_dir = os.path.join(output_base, split, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in os.listdir(in_img_dir):
        if file.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(in_img_dir, file)
            lbl_path = os.path.join(in_lbl_dir, file.rsplit(".", 1)[0] + ".txt")
            tile_image_left_right_with_overlap(img_path, lbl_path, out_img_dir, out_lbl_dir, overlap_pct)

    print(f"Finished tiling '{split}' into left-right parts with overlap.")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        tile_split_left_right(split, input_base="path/dataset", output_base="path/dataset", overlap_pct=0.01)
