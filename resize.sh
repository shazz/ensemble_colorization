#! /bin/bash

for image in images/$1/*.jpg
do
    convert $image -resize 256x256\! more_images/$(basename $image)
done
