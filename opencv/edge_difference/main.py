import cv2
import numpy as np
from numpy import linalg as LA

def show_image(image):
	cv2.namedWindow('window')
	while(1):
		cv2.imshow('window', image)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
	cv2.destroyAllWindows()

def get_edge_image_difference(diff_image):
	# Normalize: transform domain from [0,255] to [0,1]
	diff_image = diff_image / 255
	return np.sum(diff_image)*1.0 / (len(diff_image)*len(diff_image[0]))

# Read images in grayscale
control =  cv2.imread("control.jpg", 0);
damaged =  cv2.imread("aligned.jpg", 0);
#control =  cv2.imread("main.jpg", 0);
#damaged =  cv2.imread("damaged.jpg", 0);


edges_control = cv2.Canny(control, 50, 50)
edges_damaged = cv2.Canny(damaged, 50, 50)

diff_image = cv2.absdiff(edges_control, edges_damaged)

show_image(diff_image)
print get_edge_image_difference(diff_image)
