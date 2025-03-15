import sys
import cv2 as cv
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt

# Function to calculate energy using Sobel filters
def calculate_energy(image):
    grayscale_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv.Sobel(grayscale_image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(grayscale_image, cv.CV_64F, 0, 1, ksize=3)
    return np.abs(grad_x) + np.abs(grad_y)

# Dynamic programming approach to find the minimum seam
def find_min_seam(energy_map):
    rows, cols = energy_map.shape
    cumulative_energy = energy_map.copy()
    backtrack = np.zeros_like(cumulative_energy, dtype=np.int32)
    for row in range(1, rows):
        for col in range(cols):
            left = cumulative_energy[row-1, col-1] if col > 0 else float('inf')
            up = cumulative_energy[row-1, col]
            right = cumulative_energy[row-1, col+1] if col < cols-1 else float('inf')
            min_value = min(left, up, right)
            backtrack[row, col] = col - 1 if min_value == left else col + 1 if min_value == right else col
            cumulative_energy[row, col] += min_value
    return cumulative_energy, backtrack

# Function to remove a column seam and mark it on the image
def remove_column_seam(image):
    energy_map = calculate_energy(image)
    cumulative_energy, backtrack = find_min_seam(energy_map)
    mask = np.ones(image.shape[:2], dtype=bool)
    seam_path = []
    col = np.argmin(cumulative_energy[-1])
    for row in reversed(range(image.shape[0])):
        mask[row, col] = False
        seam_path.append((row, col))
        col = backtrack[row, col]
    
    # Mark the seams on the original image for visualization
    marked_image = image.copy()
    for row, col in seam_path:
        marked_image[row, col] = [255, 0, 0]  # Red color for marked seam
    
    # Return the image with the seam removed
    return image[mask].reshape((image.shape[0], image.shape[1] - 1, 3)), marked_image

# Function to perform seam carving for resizing the image
def seam_carve_resize(image, scale_factor):
    new_width = int(scale_factor * image.shape[1])
    marked_image = image.copy()
    for _ in range(image.shape[1] - new_width):
        image, marked_image = remove_column_seam(image)
    return image, marked_image

# Main function to execute the seam carving process
def main(image_path, scale_factor=0.9, output_path="output_image.jpg"):
    input_image = imread(image_path)
    resized_image, marked_image = seam_carve_resize(input_image, scale_factor)
    imwrite(output_path, resized_image)
    imwrite("highlighted_seams.jpg", marked_image)
    
    # Display the original and resized images with seam highlights
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(marked_image)
    ax[0].set_title("Seams Highlighted")
    ax[0].axis("off")
    
    ax[1].imshow(resized_image)
    ax[1].set_title("Resized Image")
    ax[1].axis("off")
    
    plt.show()

# Command-line interface to run the script
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python file.py <image_path> <scale_factor>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]))
