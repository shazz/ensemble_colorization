#!/usr/bin/env bash
#
# color_sort.sh
#
# Sorts all of the JPEG images in the given directory into bins by their
# dominant color. The dominant color is determined by the color channel that has
# the maximum summed value in the image. This is done for each RGB color
# channel. Creates 'red', 'green', and 'blue' directories in the given output
# directory for each channel.

# Exit the script on error or an undefined variable
set -e
set -u
set -o pipefail

# Program usage, and number of images to batch process
USAGE="color_sort.sh <image_dir> <output_dir>"
IMAGES_BLOCK_SIZE=1000

# Check that the number of command line arguments is valid
num_args=$#
if [ ${num_args} -ne 2 ]; then
    printf "Error: Improper number of command line arguments.\n"
    printf "${USAGE}\n"
    exit 1
fi

# Parse the command line arguments
image_dir=$1
output_dir=$2

# Create the output directories for red, green, and blue
mkdir -p ${output_dir}/red
mkdir -p ${output_dir}/blue
mkdir -p ${output_dir}/green
mkdir -p ${output_dir}/blue_green

# Convert the images in parallel, splitting the work among 2*num_cores processes
num_cores=$(getconf _NPROCESSORS_ONLN)
num_jobs=$((2 * ${num_cores}))
find -L ${image_dir} -type f -name '*.jpg' | parallel --progress \
        -j ${num_jobs} -N ${IMAGES_BLOCK_SIZE} \
        python color_sort.py ${output_dir} {}
