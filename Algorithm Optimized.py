import sys
import numpy as np
from imageio.v2 import imread, imwrite
import matplotlib.pyplot as plt

# Compute energy using finite differences (without predefined functions)
def calculate_energy(image):
    grayscale = np.dot(image[..., :3], [0.2989, 0.587, 0.114])  # Convert to grayscale
    
    # Compute |∂I/∂x| using forward difference
    grad_x = np.abs(np.diff(grayscale, axis=1, append=grayscale[:, -1:]))
    
    # Compute |∂I/∂y| using forward difference
    grad_y = np.abs(np.diff(grayscale, axis=0, append=grayscale[-1:, :]))
    
    return grad_x + grad_y

# Optimized seam finding using NumPy vectorization
def find_min_seam(energy_map):
    rows, cols = energy_map.shape
    cumulative_energy = energy_map.copy()
    backtrack = np.zeros((rows, cols), dtype=np.int32)

    for row in range(1, rows):
        left = np.roll(cumulative_energy[row - 1], shift=1, axis=0)
        right = np.roll(cumulative_energy[row - 1], shift=-1, axis=0)
        left[0] = right[-1] = np.inf  # Boundary conditions

        min_idx = np.argmin(np.stack([left, cumulative_energy[row - 1], right]), axis=0)
        cumulative_energy[row] += np.choose(min_idx, [left, cumulative_energy[row - 1], right])
        backtrack[row] = np.choose(min_idx, [np.arange(cols) - 1, np.arange(cols), np.arange(cols) + 1])

    # Trace back the optimal seam path
    seam_path = np.zeros(rows, dtype=np.int32)
    seam_path[-1] = np.argmin(cumulative_energy[-1])

    for row in range(rows - 2, -1, -1):
        seam_path[row] = backtrack[row + 1, seam_path[row + 1]]

    return seam_path

# Efficiently remove a single seam
def remove_column_seam(image, seam_mask, seam_path):
    rows, cols, _ = image.shape
    mask = np.ones((rows, cols), dtype=bool)

    mask[np.arange(rows), seam_path] = False
    seam_mask[np.arange(rows), seam_path] = [255, 0, 0]  # Highlight seam in red

    return image[mask].reshape((rows, cols - 1, 3))

# Seam carving function with batch seam removal
def seam_carve_resize(image, scale_factor_width):
    new_width = int(image.shape[1] * scale_factor_width)
    num_seams = image.shape[1] - new_width
    seam_mask = image.copy()

    for _ in range(num_seams):
        energy_map = calculate_energy(image)
        seam_path = find_min_seam(energy_map)
        image = remove_column_seam(image, seam_mask, seam_path)

    return image, seam_mask

# Main function
def main(image_path, scale_factor_width, output_path="output_image.jpg"):
    input_image = imread(image_path)
    resized_image, marked_image = seam_carve_resize(input_image, scale_factor_width)

    imwrite(output_path, resized_image)
    imwrite("highlighted_seams.jpg", marked_image)

    # Display the original and resized images with seam highlights
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(marked_image)
    ax[0].set_title("All Seams Highlighted")
    ax[0].axis("off")

    ax[1].imshow(resized_image)
    ax[1].set_title("Resized Image")
    ax[1].axis("off")

    plt.show()

# Command-line interface
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python file.py <image_path> <scale_factor_width>")
        sys.exit(1)

    main(sys.argv[1], float(sys.argv[2]))
