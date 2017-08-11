#!/usr/bin/env python

# Goal: 
# Input: control and query image
# Output: aligned query image

import numpy as np
import cv2
import sys, getopt

import util
import draw

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6

def get_detector_and_matcher(feature_name):
	chunks = feature_name.split("-")
	if chunks[0] == "sift":
		detector = cv2.SIFT()
		norm = cv2.NORM_L2
	elif chunks[0] == "surf":
		detector = cv2.SURF(800)
		norm = cv2.NORM_L2
	elif chunks[0] == "orb":
		detector = cv2.ORB(400)
		norm = cv2.NORM_HAMMING
	else:
		detector = cv2.SIFT()
		norm = cv2.NORM_L2

	if "flann" in chunks:
		if norm == cv2.NORM_L2:
			flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		else:
			flann_params= dict(algorithm = FLANN_INDEX_LSH,
							   table_number = 6, # 12
							   key_size = 12,     # 20
							   multi_probe_level = 1) #2
		# bug : need to pass empty dict (#1329)
		matcher = cv2.FlannBasedMatcher(flann_params, {})
	else:
		matcher = cv2.BFMatcher(norm)
	return detector, matcher

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
	query_rectangle = util.get_corners(corners.reshape(4,2))
	max_h, max_w = img_control.shape[:2]

	empty_array = np.array([
		[0, 0],
		[max_w-1, 0],
		[max_w-1, max_h-1],
		[0, max_h-1]], dtype="float32")

	transform_matrix = cv2.getPerspectiveTransform(query_rectangle, empty_array)
	return cv2.warpPerspective(img_query, transform_matrix, (max_w, max_h))

def get_warped_image(img_control, img_query, draw_traces=False):
	img_control = resize(img_control)
	img_query = resize(img_query)

	features = ["sift", "surf"]
	feature_name = features[0]
	detector, matcher = get_detector_and_matcher(feature_name)

	# Find keypoints and descriptors
	kp1, desc1 = detector.detectAndCompute(img_control, None)
	kp2, desc2 = detector.detectAndCompute(img_query, None)

	# Find keypoints that match between the two images
	raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

	# Filter out weak matches
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

	if len(p1) < 4:
		print "Not enough strong matches found"
		exit()
	
	# Find the transformation homography
	# H - 3x3 transformation matrix
	# status - vector of [0,1] values (one-hots of something?)
	# Basic idea: img1 = H*img2
	H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
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
		pass
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
	img_control = cv2.imread("../img/sample_control.jpg")
	#img_query = cv2.imread("../img/sample_misaligned_undamaged.jpg")
	img_query = cv2.imread("../img/background/05.jpg")

	img_control = resize(img_control)
	img_query = resize(img_query)
	
	warped_image = get_warped_image(img_control, img_query, True)
	cv2.imwrite("../img/sample_aligned_undamaged.jpg", warped_image);

	cv2.imshow("warped", warped_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()