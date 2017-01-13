#!/usr/bin/env python
#
# color_sort.py
#
# Sorts the given image into the 'red', 'green', or 'blue' directory in the
# specified output directory based on what the dominant color is in the image.
# The dominant color is the color that has the greatest sum in the image.

import os
import numpy as np
import scipy.misc
from argparse import ArgumentParser

# The directories for red, blue, and green images
RED_DIR = "red"
BLUE_DIR = "blue"
GREEN_DIR = "green"
BLUE_GREEN_DIR = "blue_green"

# Function to check if the given pixel is grayscale (all color channels equal)
def pixel_is_grayscale(pixel):
    return pixel[0] == pixel[1] and pixel[1] == pixel[2]

# Function to check if an image is grayscale (all color channels are equal)
def image_is_grayscale(image):
    for row in image:
        for pixel in row:
            if not pixel_is_grayscale(pixel):
                return False
    return True

# Parse the command line arguments
parser = ArgumentParser(description="Sorts the given input image into the "
        "red, green, or blue directory output directory based on its dominant "
        "color. The image is copied.")
parser.add_argument("output_dir", type=str, help="The output directory to copy "
        "the image file into. It is sorted to either a 'red', 'blue', or "
        "'green' subdirectory in that directory.")
parser.add_argument("image_path", type=str, nargs='+', help="The path(s) to "
        "the image file(s) to sort based on its color.")
args = parser.parse_args()

# Create the output directory if it does not exist
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Iterate over all the image paths specified by the user
for image_path in args.image_path:
    # Load the image from file, and get the basename
    image = scipy.misc.imread(image_path)
    image_name = os.path.basename(image_path)

    # Sum the image along each color channel, skip grayscale images
    color_sums = np.sum(image, (0, 1))
    if color_sums.shape == () or image_is_grayscale(image):
        continue

    # Determine the channel with the max sum value
    max_color = np.amax(color_sums)
    red_sum = color_sums[0]
    green_sum = color_sums[1]
    blue_sum = color_sums[2]
    blue_green_sum = (green_sum + blue_sum)/2

    if blue_sum == green_sum:
        blue_sum = 0
        green_sum = 0

    # Select the output directory to save to based on the max color. If there
    # are multiple that match, randomly select one of them
    color_pairs = [(red_sum, RED_DIR), (blue_sum, BLUE_DIR), (green_sum, GREEN_DIR), (blue_green_sum, BLUE_GREEN_DIR)]
    max_colors = [os.path.join(args.output_dir, color) for (color_sum, color) in color_pairs if color_sum == max_color]
    output_image_dir = np.random.choice(max_colors)

    # Create the output directory if needed, and save the output image
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    output_image_path = os.path.join(output_image_dir, image_name)
    scipy.misc.imsave(output_image_path, image)
