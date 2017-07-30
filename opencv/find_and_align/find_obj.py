#!/usr/bin/env python

''' From docs:
Feature-based image matching sample.

USAGE
  find_obj.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
				to use Flann-based matcher instead bruteforce.

'''

# Goal: 
# Input: control and query image
# Output: aligned and cropped query image

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
		fn2 = 'rotated.jpg'
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
	#print "Detector:", detector
	#print "Matcher:", matcher
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


def draw_match(window, img1, img2, kp_pairs, status=None):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	img_compare = np.zeros((max(h1, h2), w1+w2), np.uint8)
	img_compare[:h1, :w1] = img1
	img_compare[:h2, w1:w1+w2] = img2
	img_compare = cv2.cvtColor(img_compare, cv2.COLOR_GRAY2BGR)

	blank_array = np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(2,-1,2)
	corners = cv2.perspectiveTransform(blank_array, H)
	
	# TODO
	#---------------
	# Determine topleft,topright,bottomright,bottomleft corners
	points = corners.reshape(4,2)
	query_rectangle = get_corners(points)
	print query_rectangle
	max_width, max_height = get_width_height(query_rectangle)
	print max_width, max_height

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
	warp = cv2.warpPerspective(img2, M, (max_width, max_height))

	cv2.imshow("image", warp)
	cv2.waitKey(0)

	#---------------

	warp = cv2.warpPerspective(img_compare, H, (w1, h1))

	offset_corners = np.int32(corners.reshape(-1, 2) + (w1, 0))
	draw_match_rect(img_compare, [offset_corners])

	if status is None:
		status = np.ones(len(kp_pairs), np.bool_)
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	# Draw inliers (if successfully found the object)
	# or matching keypoints (if failed)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			draw_match_line(img_compare, x1, x2, y1, y2, 2)
		else:
			draw_cross(img_compare, x1, x2, y1, y2, 2)
	
	display_image("original", img_compare)
	display_image("aligned", warp)


if __name__ == '__main__':
	fn1, fn2, feature_name = parse_args()
	img1 = cv2.imread(fn1, 0)
	img2 = cv2.imread(fn2, 0)
	#print feature_name

	detector, matcher = get_detector_and_matcher(feature_name)

	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)
	#print "img1 - %d features" % len(kp1)
	#print "img2 - %d features" % len(kp2)

	raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2) #2
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
	if len(p1) >= 4:
		# H - 3x3 transformation matrix (homography)
		# status - [0,1] vector of ???
		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
		#print '%d / %d  inliers/matched' % (np.sum(status), len(status))
	else:
		H, status = None, None
		#print '%d matches found, not enough for homography estimation' % len(p1)

	if H is not None:
		draw_match("find_obj", img1, img2, kp_pairs, status)

	cv2.waitKey()
	cv2.destroyAllWindows()
