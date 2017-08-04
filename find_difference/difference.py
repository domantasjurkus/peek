import cv2
import numpy as np
from numpy import linalg as LA

def get_quality_score(img_diff):
	# Normalize: transform domain from [0,255] to [0,1]
	difference_score = np.sum(img_diff)*1.0 / (len(img_diff)*len(img_diff[0]))
	return 100 - difference_score

def get_difference_image(img1, img2):
	# TODO: figure out if auto-setting blur parameters is possible
	img1 = cv2.GaussianBlur(img1, (5,5), 0)

	'''
	edges_control = cv2.Canny(img1, 50, 50)
	edges_damaged = cv2.Canny(img2, 50, 50)
	diff = cv2.absdiff(edges_control, edges_damaged)
	'''

	diff = cv2.absdiff(img1, img2)
	return diff

if __name__ == "__main__":
	img1_path = "../img/control.jpg"
	img2_path = "../img/aligned_paper.jpg"
	img1 = cv2.imread(img1_path, 0);
	img2 = cv2.imread(img2_path, 0);

	img_diff = get_difference_image(img1, img2)
	damage_prob = get_quality_score(img_diff)
	print damage_prob

	cv2.imshow("original", img1)
	cv2.imshow("damaged", img2)
	cv2.imshow("comparison", img_diff)
	cv2.waitKey(0)
