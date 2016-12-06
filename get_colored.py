#! /usr/bin/python

import os
from scipy import misc
import numpy as np

dir = 'rgb_imgs/'
red_dir = 'red/'
blue_dir = 'blue/'
green_dir = 'green/'

i = 0
for f in os.listdir(dir):
    try:
        image = misc.imread(dir + f)
        
        s = np.sum(image, (0,1))
        
        m = np.amax(s)
        red = s[0]
        green = s[1]
        blue = s[2]
        
        if (red == m):
            misc.imsave(red_dir + f, image)
        elif (blue == m):
            misc.imsave(blue_dir + f, image)
        elif (green == m):
            misc.imsave(green_dir + f, image)
        else:
            print s

        if i % 100 == 0:
            print 'Done image %d!' % i

        i += 1

    except:
        continue
