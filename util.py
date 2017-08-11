import cv2

def resize(img, maximum_small_edge=500):
	h = img.shape[0]
	w = img.shape[1]
	small_edge = h if h < w else w

	# If the image is already 500px or smaller on the shorter edge
	if small_edge <= maximum_small_edge:
		return img

	scale_ratio = 1 / (small_edge*1.0 / maximum_small_edge)
	return cv2.resize(img, (0,0), fx=scale_ratio, fy=scale_ratio)