import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def explore_match(window, img1, img2, kp_pairs, status = None, H = None):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	if H is not None:
		corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
		corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
		cv2.polylines(vis, [corners], True, (255, 255, 255))

	if status is None:
		status = np.ones(len(kp_pairs), np.bool_)
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	green = (0, 255, 0)
	red = (0, 0, 255)
	white = (255, 255, 255)
	kp_color = (51, 103, 236)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			col = green
			cv2.circle(vis, (x1, y1), 2, col, -1)
			cv2.circle(vis, (x2, y2), 2, col, -1)
		else:
			col = red
			r = 2
			thickness = 3
			cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
			cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
			cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
			cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
	vis0 = vis.copy()
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			cv2.line(vis, (x1, y1), (x2, y2), green)

	cv2.imshow(window, vis)
	'''def onmouse(event, x, y, flags, param):
		cur_vis = vis
		if flags & cv2.EVENT_FLAG_LBUTTON:
			cur_vis = vis0.copy()
			r = 8
			m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
			idxs = np.where(m)[0]
			kp1s, kp2s = [], []
			for i in idxs:
				 (x1, y1), (x2, y2) = p1[i], p2[i]
				 col = (red, green)[status[i]]
				 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
				 kp1, kp2 = kp_pairs[i]
				 kp1s.append(kp1)
				 kp2s.append(kp2)
			cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
			cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

		cv2.imshow(win, cur_vis)
	cv2.setMouseCallback(win, onmouse)
	return vis'''


def filter_matches(kp1, kp2, matches, ratio=0.75):
	# Matched key points
	mkp1, mkp2 = [], []



def main():
	img1 = cv2.imread("c1.jpg",0)
	img2 = cv2.imread("c2.jpg",0)

	# Find the keypoints and descriptors with SIFT
	detector = cv2.SIFT()
	kp1, des1 = detector.detectAndCompute(img1, None)
	kp2, des2 = detector.detectAndCompute(img2, None)

	# Matching
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	matcher = cv2.FlannBasedMatcher(index_params, search_params)
	raw_matches = matcher.knnMatch(des1, des2, k=2)

	# Store all the good matches as per Lowe's ratio test
	good = []
	for m,n in raw_matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	if len(good) < MIN_MATCH_COUNT:
		print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
		matchesMask = None
		exit()

	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	kp_pairs = zip(src_pts, dst_pts);

	# The magic
	H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	#matchesMask = mask.ravel().tolist()

	h,w = img1.shape
	
	#pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
	#dst = cv2.perspectiveTransform(pts, H)
	#something = cv2.polylines(img2, [np.int32(dst)], True, 255, 3)

	'''draw_params = dict(matchColor=(0,255,0), # draw matches in green color
				   singlePointColor=None,
				   matchesMask=matchesMask, # draw only inliers
				   flags=2)'''


	print "kp_pairs:", kp_pairs[0]
	explore_match("main_window", img1, img2, kp_pairs, status, H)
	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	#plt.imshow(img3, 'gray')
	#plt.show()

if __name__ == "__main__":
	main()
