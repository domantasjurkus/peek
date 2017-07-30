#!/usr/bin/env python

# Goal: 
# Input: control and query image
# Output: aligned query image

import numpy as np
import cv2
import sys, getopt

from draw import *
from util import *

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def parse_args():
	opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
	opts = dict(opts)
	#feature_name = opts.get('--feature', 'surf')
	feature_name = opts.get('--feature', 'sift')
	try:
		fn1, fn2 = args
	except:
		fn1 = 'control.jpg'
		fn2 = 'damaged.jpg'
	return fn1, fn2, feature_name


def get_detector_and_matcher(feature_name):
	chunks = feature_name.split('-')
	if chunks[0] == 'sift':
		detector = cv2.SIFT()
		norm = cv2.NORM_L2
	elif chunks[0] == 'surf':
		detector = cv2.SURF(800)
		norm = cv2.NORM_L2
	elif chunks[0] == 'orb':
		detector = cv2.ORB(400)
		norm = cv2.NORM_HAMMING
	else:
		detector = cv2.SIFT()
		norm = cv2.NORM_L2

	if 'flann' in chunks:
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


def find_in_query_image(img_control, img_query):
	# WORK IN PROGRESS
	# Determine topleft,topright,bottomright,bottomleft corners
	points = corners.reshape(4,2)
	query_rectangle = get_corners(points)
	max_height, max_width = img_control.shape[:2]

	# Create a destination array to
	# map the screen to a top-down, "birds eye" view
	dst = np.array([
		[0, 0],
		[max_width-1, 0],
		[max_width-1, max_height-1],
		[0, max_height-1]], dtype="float32")

	# Calculate the perspective transform matrix
	M = cv2.getPerspectiveTransform(query_rectangle, dst)

	# Warp the perspective to grab the screen
	warp = cv2.warpPerspective(img_query, M, (max_width, max_height))
	cv2.imshow("image", warp)


if __name__ == '__main__':
	fn1, fn2, feature_name = parse_args()
	img_control = cv2.imread(fn1, 0)
	img_query = cv2.imread(fn2, 0)

	detector, matcher = get_detector_and_matcher(feature_name)

	kp1, desc1 = detector.detectAndCompute(img_control, None)
	kp2, desc2 = detector.detectAndCompute(img_query, None)

	raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

	if len(p1) < 4:
		print "Not enough matches found"
		exit()
	
	# H - 3x3 transformation matrix (homography)
	# status - [0,1] vector of ???
	H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	#print '%d / %d  inliers/matched' % (np.sum(status), len(status))

	if H is None:
		print "Homography could not be found"
		exit()

	h1, w1 = img_control.shape[:2]
	h2, w2 = img_query.shape[:2]
	img_combine = np.zeros((max(h1, h2), w1+w2), np.uint8)
	img_combine[:h1, :w1] = img_control
	img_combine[:h2, w1:w1+w2] = img_query
	img_combine = cv2.cvtColor(img_combine, cv2.COLOR_GRAY2BGR)

	blank_array = np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(2,-1,2)
	corners = cv2.perspectiveTransform(blank_array, H)

	draw_traces(img_control, img_query, kp_pairs, corners, status)
	find_in_query_image(img_control, img_query)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
