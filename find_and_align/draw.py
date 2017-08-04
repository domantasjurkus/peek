import numpy as np
import cv2

def draw_match_rect(image, corners, color=(255,255,255)):
	cv2.polylines(image, corners, True, color)


def draw_match_line(image, x1, x2, y1, y2, r, color=(0,255,0)):
	cv2.circle(image, (x1, y1), r, color, -1)
	cv2.circle(image, (x2, y2), r, color, -1)
	cv2.line(image, (x1, y1), (x2, y2), color)


def draw_cross(image, x1, x2, y1, y2, r, color=(0,0,255), thickness=3):
	cv2.line(image, (x1-r, y1-r), (x1+r, y1+r), color, thickness)
	cv2.line(image, (x1-r, y1+r), (x1+r, y1-r), color, thickness)
	cv2.line(image, (x2-r, y2-r), (x2+r, y2+r), color, thickness)
	cv2.line(image, (x2-r, y2+r), (x2+r, y2-r), color, thickness)


def draw_traces(img_control, img_query, kp_pairs, corners, status=None):
	h1, w1 = img_control.shape[:2]
	h2, w2 = img_query.shape[:2]
	img_combine = np.zeros((max(h1, h2), w1+w2), np.uint8)
	img_combine[:h1, :w1] = img_control
	img_combine[:h2, w1:w1+w2] = img_query
	img_combine = cv2.cvtColor(img_combine, cv2.COLOR_GRAY2BGR)

	offset_corners = np.int32(corners.reshape(-1, 2) + (w1, 0))
	draw_match_rect(img_combine, [offset_corners])

	if status is None:
		status = np.ones(len(kp_pairs), np.bool_)
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	# Draw inliers (if successfully found the object)
	# or matching keypoints (if failed)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			draw_match_line(img_combine, x1, x2, y1, y2, 2)
		else:
			draw_cross(img_combine, x1, x2, y1, y2, 2)
	
	cv2.imshow("original", img_combine)
