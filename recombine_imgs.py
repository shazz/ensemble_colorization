import numpy as np
import sys



def main():
	args = sys.argv

	if(len(args) < 2):
		print("Usage: ./recombine_imgs.py <channel_name_file>") 
		return

	chanFile = args[1]

	#map of directory name to weight of channel 
	chanDict = {}

	#Get channels
	with open(chanFile,'r') as f:
		for line in f:
			line = line.rstrip

			params = line.split()

			#Populate the dict
			if len(params) > 1:
				chanDict[params[0]] = float(params[1])

	#Check if we have valid channels
	if len(chanDict) < 1:
		print("No channels present")
		return


def recombineImage(channelList,weights,imgIndex,row,col):

	result = np.zeros(3)#Set rgb to 0

	numChans = len(channelList)

	currMax = 0
	maxIndex = 0

	#First get max intensity
	for i in xrange(numChans):
		#Get rgb value
		pixel = channelList[i][imgIndex][row][col]

		maxIntensity = np.amax(pixel)

		if maxIntensity > currMax:
			maxIndex = i
			currMax = maxIntensity

	for i in xrange(numChans):
		pixel = channelList[i][imgIndex][row][col]
		if i == maxIndex:
			addVal = pixel
		else: 
			addVal = np.multiply(weights[i],pixel)

		result = np.add(result,addVal)

	return result

def doRecombine(channelList,weights):
	#The channelList contains a list of channels
	#Each channel contains a list of images in the channel

	chanDims = channelList.shape

	lenImgs = chanDims[1]

	imgRows = chanDims[2]

	imgCols = chanDims[3]

	result = np.zeros((lenImgs,imgRows,imgCols,3))

	#For each image
	for i in xrange(lenImgs):
		for j in xrange(imgRows):
			for k in xrange(imgCols):
				#For each pixel in the image
				newPixel = recombineImage(channelList,weights,i,j,k)

				for l in xrange(3):
					result[i][j][k][l] = newPixel[l]

	return result



if __name__ == "__main__":
    main()