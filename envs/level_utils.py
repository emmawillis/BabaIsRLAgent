import shutil
import cv2
import numpy as np
import os
import json
from .game_objects import Object

image_to_key = {
    "baba": Object.BABA,
    "wall": Object.WALL,
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

def split_into_grid(image, grid_size=(17, 15)):
    """Splits an image into a 33x18 grid of cells."""
    h, w = image.shape[:2]
    cell_width = w // grid_size[0]
    cell_height = h // grid_size[1]

    grid_cells = []

    for y in range(grid_size[1]):
        row = []
        for x in range(grid_size[0]):
            x_start, y_start = x * cell_width, y * cell_height
            x_end, y_end = (x + 1) * cell_width, (y + 1) * cell_height
            cell = image[y_start:y_end, x_start:x_end]
            grid_cells.append(((x, y), cell))
        
    return grid_cells, grid_size


def match_objects(grid_cells, grid_size, reference_images, threshold=0.35):
    """Matches each grid cell to the closest reference image and returns a 2D array representation."""
    grid = [[0 for _ in range(grid_size[0])] for _ in range(grid_size[1])]

    for (x, y), cell in grid_cells:
        best_match = None
        best_score = -1

        # Convert grid cell to grayscale
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        for obj_name, ref_img in reference_images.items():
            if ref_img.shape[-1] == 4:
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2BGR)
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
        # else:
        #     # Show unmatched cell
        #     cv2.imshow(f"Unmatched ({x}, {y})", cell)
        #     cv2.waitKey(0)  # Wait for key press before closing
        #     cv2.destroyAllWindows()  # Close the window

    return grid

def load_reference_images(reference_folder):
    """Loads all reference object images from a folder."""
    reference_images = {}
    for filename in os.listdir(reference_folder):
        if filename.endswith(".png") and not filename.startswith("image"):  # Exclude hidden files
            obj_name = os.path.splitext(filename)[0]  # Use filename as object label
            img = cv2.imread(os.path.join(reference_folder, filename), cv2.IMREAD_UNCHANGED)
            if obj_name in ["WALL-1", "WALL-2", "WALL-3"]:
                obj_name = "WALL"
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
