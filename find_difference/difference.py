import os
import cv2
import numpy as np

def get_quality_score(img_diff):
	# Normalize: transform domain from [0,255] to [0,1]
	pixel_count = len(img_diff)*len(img_diff[0])
	difference_score = np.sum(img_diff)*1.0 / pixel_count
	return 100 - difference_score

def get_absolute_image_difference(img1, img2):
	"""
	Horrible horrible method - takes background and the slightest misalignment into account
	"""
	return cv2.absdiff(img1, img2)

def feed_background_images(subtractor, directory="../img/background/aligned_fixed/"):
	"""Experimental feature that does not work"""
	for filename in os.listdir(directory):
		if filename[-4:] != ".jpg":
			continue
		filepath = directory+filename
		background_image = cv2.imread(filepath)
		temp = subtractor.apply(background_image, learningRate=0.5)
		print "Added %s to subtractor" % filepath

def get_foreground_mask(img_query):
	"""Experimental feature that does not work"""
	subtractor = cv2.BackgroundSubtractorMOG2()
	feed_background_images(subtractor)

	foreground_mask = subtractor.apply(img2, learningRate=0)
	return foreground_mask

def get_difference_image(img1, img2, show_images=False):
	#diff = get_foreground_mask(img2)
	
	cv2.GaussianBlur(img1, (5,5), 10, img1)
	cv2.GaussianBlur(img2, (5,5), 10, img2)

	diff = get_absolute_image_difference(img1, img2)
	cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY, diff)
	#cv2.adaptiveThreshold(diff, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
	#	thresholdType=cv2.THRESH_BINARY_INV, blockSize=3, C=4, dst=diff) 

	if show_images:
		cv2.imshow("control", img1)
		cv2.imshow("query", img2)
		cv2.imshow("difference", diff)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return diff

if __name__ == "__main__":
	img1_path = "../img/sample_control.jpg"
	img2_path = "../img/sample_aligned_damaged.jpg"
	#img2_path = "../img/sample_aligned_undamaged.jpg"
	img1 = cv2.imread(img1_path, 0);
	img2 = cv2.imread(img2_path, 0);

	img_diff = get_difference_image(img1, img2, True)

	damage_prob = get_quality_score(img_diff)
	print damage_prob

