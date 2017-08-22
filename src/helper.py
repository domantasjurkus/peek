import numpy as np

def get_corners(points):
	# Input: array of 4 coordinates [x,y]
	# Returns: [top-left,top-right,bottom-right,bottom-left]

	# rect = [top-left,top-right,bottom-right,bottom-left]
	rect = np.zeros((4, 2), dtype="float32")

	# Find top-left and bottom-right corners
	# Top-left has smallest sum
	# Bottom-right has largest sum
	s = points.sum(axis=1)
	rect[0] = points[np.argmin(s)]
	rect[2] = points[np.argmax(s)]

	# Find the top-right and bottom-left corners
	# Top-right: difference from x to y will be smallest
	# Bottom-left: difference from x to y will be smallest
	diff = np.diff(points, axis=1)
	rect[1] = points[np.argmin(diff)]
	rect[3] = points[np.argmax(diff)]

	return rect


def get_width_height(quad_corners):
	# Input: [top-left,top-right,bottom-right,bottom-left]
	# Output: (maximum_width,maximum_height)
	(tl, tr, br, bl) = quad_corners

	width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

	height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	max_width = max(int(width_top), int(width_bottom))
	max_height = max(int(height_left), int(height_right))

	return (max_width, max_height)
