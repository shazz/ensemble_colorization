#!/usr/bin/env /bin/bash
#
# resize.sh
#
# Resizes the all the JPG images in the given directory to the required size
# for the recolorization CNN, 256x256.

# Exit the script on error or an undefined variable
set -e
set -u
set -o pipefail

# Program usage, number of parallel processes to spawn, and output image size
USAGE="./resize.sh <image_dir> <output_dir>"
IMAGE_SIZE=256x256

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

# Make the output directory if it doesn't exist
mkdir -p ${output_dir}

# Convert images in parallel, using as many processes as possible
num_cores=$(getconf _NPROCESSORS_ONLN)
num_jobs=$((2 * ${num_cores}))
find -L ${image_dir} -type f -name '*.jpg' | parallel -j${num_jobs} --progress \
        convert -resize ${IMAGE_SIZE}\! {} ${output_dir}/{/}
