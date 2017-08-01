import cv2
import numpy as np
from numpy import linalg as LA

def get_edge_image_difference(diff_image):
	# Normalize: transform domain from [0,255] to [0,1]
	return np.sum(diff_image)*1.0 / (len(diff_image)*len(diff_image[0]))

# Read images in grayscale
img1 =  cv2.imread("control.jpg", 0);
img2 =  cv2.imread("paper.jpg", 0);
#control =  cv2.imread("main.jpg", 0);
#damaged =  cv2.imread("damaged.jpg", 0);

img1 = cv2.GaussianBlur(img1, (5,5), 0)

test = cv2.absdiff(img1, img2)

edges_control = cv2.Canny(img1, 50, 50)
edges_damaged = cv2.Canny(img2, 50, 50)

#sum = np.diff(img1, img2)

diff_image = cv2.absdiff(edges_control, edges_damaged)

print get_edge_image_difference(test)

cv2.imshow("original", img1)
cv2.imshow("damaged", img2)
cv2.imshow("comparison", test)
cv2.waitKey(0)