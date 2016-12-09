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

    # Sum the image along each color channel, and determine the max sum value
    color_sums = np.sum(image, (0, 1))
    max_color = np.amax(color_sums)
    red_max = color_sums[0]
    green_max = color_sums[1]
    blue_max = color_sums[2]

    # Select the output directory to save too based on the max color
    if red_max == max_color:
        output_image_dir = os.path.join(args.output_dir, RED_DIR)
    elif green_max == max_color:
        output_image_dir = os.path.join(args.output_dir, GREEN_DIR)
    else:
        output_image_dir = os.path.join(args.output_dir, BLUE_DIR)

    # Create the output directory if needed, and save the output image
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    output_image_path = os.path.join(output_image_dir, image_name)
    scipy.misc.imsave(output_image_path, image)
