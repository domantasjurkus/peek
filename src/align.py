import numpy as np
import cv2
import sys, os

import helper
import draw

def filter_matches(kp1, kp2, matches, ratio=0.75):
	# matched key points
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append( kp1[m.queryIdx] )
			mkp2.append( kp2[m.trainIdx] )
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	return p1, p2, kp_pairs


def compute_perspective(img_control, img_query, corners):
	# Determine topleft,topright,bottomright,bottomleft corners
	query_rectangle = helper.get_corners(corners.reshape(4,2))
	max_h, max_w = img_control.shape[:2]

	empty_array = np.array([
		[0, 0],
		[max_w-1, 0],
		[max_w-1, max_h-1],
		[0, max_h-1]], dtype="float32")

	# Calculates a perspective transform from four pairs of points
	perspective_transform_matrix = cv2.getPerspectiveTransform(query_rectangle, empty_array)

	# Transform query image based on matrix
	return cv2.warpPerspective(img_query, perspective_transform_matrix, (max_w, max_h))


def get_warped_image(img_control, img_query, draw_traces=False):
	img_control = resize(img_control)
	img_query = resize(img_query)

	# TODO: switch from patented SIFT to free-to-use BRISK (or other)
	detector = cv2.SIFT()
	matcher = cv2.BFMatcher(cv2.NORM_L2)

	# Find keypoints and descriptors
	keypoints1, descriptors1 = detector.detectAndCompute(img_control, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img_query, None)

	# Find keypoints that match between the two images
	raw_matches = matcher.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)

	# Filter out weak matches
	points1, points2, kp_pairs = filter_matches(keypoints1, keypoints2, raw_matches)

	if len(points1) < 4:
		print "Not enough strong matches found"
		exit()
	
	# Find the transformation homography
	# H - 3x3 transformation matrix
	# status - vector of [0,1] one-hot values
	# Basic idea: img1 = H*img2
	H, status = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
	#print "%d / %d  inliers/matched" % (np.sum(status), len(status))

	if H is None:
		print "Homography could not be found"
		exit()

	# Extract corners in the query image
	h1, w1 = img_control.shape[:2]
	h2, w2 = img_query.shape[:2]
	blank_array = np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(2,-1,2)
	corners = cv2.perspectiveTransform(blank_array, H)

	if draw_traces:
		# Draw matching keypoint pairs for debugging
		# Traces can only be drawn on grayscale images
		gray_control = cv2.cvtColor(img_control, cv2.COLOR_BGR2GRAY)
		gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
		draw.draw_traces(gray_control, gray_query, kp_pairs, corners, status)

	warped = compute_perspective(img_control, img_query, corners)
	return warped


def resize(img, maximum_small_edge=500):
	h = img.shape[0]
	w = img.shape[1]
	small_edge = h if h < w else w

	# If the image is already 500px or smaller on the shorter edge
	if small_edge <= maximum_small_edge:
		return img

	scale_ratio = 1 / (small_edge*1.0 / maximum_small_edge)
	return cv2.resize(img, (0,0), fx=scale_ratio, fy=scale_ratio)


if __name__ == "__main__":
	control_path = "img/sample_control.jpg"
	query_path = "img/sample_misaligned_damaged.jpg"
	img_control = cv2.imread(control_path)
	img_query = cv2.imread(query_path)

	if img_control is None:
		print "error: cannot load query image %s - are you sure the path is right?" % control_path
		exit()

	if img_query is None:
		print "error: cannot load query image %s - are you sure the path is right?" % query_path
		exit()
	
	warped_image = get_warped_image(img_control, img_query, True)
	#cv2.imwrite("../img/sample_aligned_undamaged.jpg", warped_image);

	cv2.imshow("warped", warped_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
