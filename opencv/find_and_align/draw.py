import numpy as np
import cv2

def draw_match(window, img1, img2, kp_pairs, status=None, H=None):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	green = (0, 255, 0)
	red = (0, 0, 255)
	white = (255, 255, 255)

	if H is not None:
		blank_array = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(2, -1, 2)
		corners = cv2.perspectiveTransform(blank_array, H)

		warp = cv2.warpPerspective(vis, H, (w1, h1))

		offset_corners = np.int32(corners.reshape(-1, 2) + (w1, 0))
		draw_match_rect(vis, [offset_corners], white)

	if status is None:
		status = np.ones(len(kp_pairs), np.bool_)
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	# Draw inliers (if successfully found the object)
	# or matching keypoints (if failed)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			draw_match_line(vis, x1, x2, y1, y2, 2, green)
		else:
			draw_cross(vis, x1, x2, y1, y2, 2, red, 3)
	
	display_image("original", vis)
	display_image("aligned", warp)


def draw_match_rect(image, corners, color):
	cv2.polylines(image, corners, True, color)


def draw_match_line(image, x1, x2, y1, y2, r, color):
	cv2.circle(image, (x1, y1), r, color, -1)
	cv2.circle(image, (x2, y2), r, color, -1)
	cv2.line(image, (x1, y1), (x2, y2), color)


def draw_cross(image, x1, x2, y1, y2, r, color, thickness):
	cv2.line(image, (x1-r, y1-r), (x1+r, y1+r), color, thickness)
	cv2.line(image, (x1-r, y1+r), (x1+r, y1-r), color, thickness)
	cv2.line(image, (x2-r, y2-r), (x2+r, y2+r), color, thickness)
	cv2.line(image, (x2-r, y2+r), (x2+r, y2-r), color, thickness)


def display_image(window_name, image_array):
	cv2.imshow(window_name, image_array)