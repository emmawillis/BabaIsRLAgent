import shutil
import cv2
import numpy as np
import os
import json
from .game_objects import Object
import matplotlib.pyplot as plt

image_to_key = {
    "baba": Object.BABA,
    "baba_2": Object.BABA,
    "wall": Object.WALL,
    "wall_2": Object.WALL,
    "wall_3": Object.WALL,
    "flag": Object.FLAG,
    "rock": Object.ROCK,
    "baba_text": Object.BABA_TEXT,
    "wall_text": Object.WALL_TEXT,
    "flag_text": Object.FLAG_TEXT,
    "rock_text": Object.ROCK_TEXT,
    "push_text": Object.PUSH_TEXT,
    "stop_text": Object.STOP_TEXT,
    "you_text": Object.YOU_TEXT,
    "win_text": Object.WIN_TEXT,
    "is_text": Object.IS_TEXT,
}

def flatten_alpha_to_black(img):
    """If an image has an alpha channel, flatten it onto a black background."""
    if img is None:
        return None
    if img.shape[-1] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3]
        bg = np.zeros_like(rgb, dtype=np.uint8)  # black background
        blended = (rgb * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
        return blended
    return img  # Already BGR

def split_into_grid(image,grid_size, debug=False):
    """
    Splits the game screenshot into grid cells by detecting blue grid lines.
    Handles transparent lines, cropped edges, and filters out noise.
    Returns: (grid_cells, grid_size)
    """

    def find_lines_filtered(projection, threshold_ratio=0.4, min_spacing=5, min_tile_spacing=50):
        """Filters noisy line detections and keeps only regularly spaced grid lines."""
        max_val = np.max(projection)
        indices = np.where(projection > threshold_ratio * max_val)[0]
        lines = []
        prev = -min_spacing
        for i in indices:
            if i - prev >= min_spacing:
                lines.append(i)
                prev = i

        # Estimate most common tile spacing
        diffs = np.diff(lines)
        if len(diffs) == 0:
            return lines
        hist = np.bincount(diffs[diffs > min_tile_spacing])
        if len(hist) == 0:
            return lines
        common_spacing = np.argmax(hist)

        # Keep lines within ±10% of common spacing
        filtered = [lines[0]]
        for i in range(1, len(lines)):
            if abs(lines[i] - filtered[-1] - common_spacing) <= common_spacing * 0.2:
                filtered.append(lines[i])

        # Add boundaries if needed
        if filtered[0] > min_spacing:
            filtered = [0] + filtered
        if filtered[-1] < len(projection) - min_spacing:
            filtered.append(len(projection) - 1)

        return filtered

    # Convert to HSV and isolate the blue grid lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 50, 30])
    upper_blue = np.array([115, 255, 120])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    vertical_proj = np.sum(mask, axis=0)
    horizontal_proj = np.sum(mask, axis=1)

    vertical_lines = find_lines_filtered(vertical_proj)
    horizontal_lines = find_lines_filtered(horizontal_proj)

    # Measure tile dimensions
    tile_width = int(np.median(np.diff(vertical_lines)))
    tile_height = int(np.median(np.diff(horizontal_lines)))

    grid_cells = []
    for y in range(len(horizontal_lines) - 1):
        for x in range(len(vertical_lines) - 1):
            x_start = vertical_lines[x]
            x_end = vertical_lines[x + 1]
            y_start = horizontal_lines[y]
            y_end = horizontal_lines[y + 1]
            cell = image[y_start:y_end, x_start:x_end]
            grid_cells.append(((x, y), cell))

    grid_size = (len(vertical_lines) - 1, len(horizontal_lines) - 1)

    if debug:
        spacing = 4
        debug_img = np.full((
            grid_size[1] * (tile_height + spacing) - spacing,
            grid_size[0] * (tile_width + spacing) - spacing,
            3), 40, dtype=np.uint8)

        for (x, y), cell in grid_cells:
            y_start = y * (tile_height + spacing)
            x_start = x * (tile_width + spacing)
            resized = cv2.resize(cell, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            debug_img[y_start:y_start + tile_height, x_start:x_start + tile_width] = resized

        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Debug: Grid Cells Based on Blue Lines (filtered + edges)")
        plt.axis("off")
        plt.show()

    return grid_cells, grid_size

def match_objects(grid_cells, grid_size, reference_images, threshold=0.5, debug=False):
    """Matches each grid cell to the closest reference image and returns a 2D array representation."""
    grid = [[0 for _ in range(grid_size[0])] for _ in range(grid_size[1])]

    for (x, y), cell in grid_cells:
        best_match = None
        best_score = -1

        wall_rgb = np.array([65, 49, 41])
        wall_tolerance = 10  # +/- range for each channel

        # Fast pixel-wise check using broadcasting
        diff = np.abs(cell.astype(np.int16) - wall_rgb[None, None, :])
        mask = np.all(diff <= wall_tolerance, axis=2)
        wall_fraction = np.mean(mask)

        if wall_fraction > 0.5:
            grid[y][x] = Object.WALL.value
            continue  # skip matching

        # Convert grid cell to grayscale
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        for obj_name, ref_img in reference_images.items():
            if ref_img.shape[-1] == 4:
                ref_img = flatten_alpha_to_black(ref_img)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

            # Resize reference image to match grid cell
            ref_resized = cv2.resize(ref_gray, (cell_gray.shape[1], cell_gray.shape[0]))

            # Apply template matching
            result = cv2.matchTemplate(cell_gray, ref_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = obj_name

        if best_match and best_score > threshold:
            obj_key = image_to_key.get(best_match.lower(), Object.BACKGROUND)
            grid[y][x] = obj_key.value
        elif debug:
            # Show unmatched cell
            cv2.imshow(f"Unmatched ({x}, {y})", cell)
            cv2.waitKey(0)  # Wait for key press before closing
            cv2.destroyAllWindows()  # Close the window

    return grid

def load_reference_images(reference_folder):
    """Loads all reference object images from a folder."""
    reference_images = {}
    for filename in os.listdir(reference_folder):
        if filename.endswith(".png") and not filename.startswith("image"):  # Exclude hidden files
            obj_name = os.path.splitext(filename)[0]  # Use filename as object label
            img = cv2.imread(os.path.join(reference_folder, filename), cv2.IMREAD_UNCHANGED)
            reference_images[obj_name] = img
    return reference_images


def save_to_json(grid, output_filename="level.json"):
    """Saves the 2D object grid to a JSON file."""
    with open(output_filename, "w") as f:
        json.dump(grid, f, indent=4)


def get_level_from_screenshot(input_screenshot, grid_size, reference_folder, output_filename=None, should_save=False):
    reference_images = load_reference_images(reference_folder)

    screenshot = cv2.imread(input_screenshot)
    
    grid_cells, grid_size = split_into_grid(screenshot, grid_size=grid_size)
    detected_grid = match_objects(grid_cells, grid_size, reference_images)
    
    if should_save:
        save_to_json(detected_grid, output_filename)
        print(f"Detection results saved to {output_filename}")

    return np.array(detected_grid, dtype="uint8").T

def get_level_from_json(input_filename="level.json"):
    """Loads the 2D object grid from a JSON file and returns it as a NumPy array."""
    with open(input_filename, "r") as f:
        grid = json.load(f)
    return np.array(grid, dtype=np.int32).T
