#!/usr/bin/env bash
#
# crop_summary_images.sh
#
# Crops the summary images from the given directory, getting back the original
# image from the summary image. A summary image is the grayscale, recolored,
# and original images concatenated together.

# Exit the script on error or an undefined variable
set -e
set -u
set -o pipefail

# Program usage, and the crop to use to extract the original image
USAGE="crop_summary_images.sh <image_dir> <output_dir>"
ORIGINAL_CROP="224x224+448+0"

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

# Create the output directory, if it does not exist
mkdir -p ${output_dir}

# Crop the images in parallel, splitting the work among 2*num_cores processes
num_cores=$(getconf _NPROCESSORS_ONLN)
num_jobs=$((2 * ${num_cores}))
find -L ${image_dir} -type f | parallel --progress -j ${num_jobs} \
        convert -crop ${ORIGINAL_CROP} +repage {} ${output_dir}/{/.}.jpg
