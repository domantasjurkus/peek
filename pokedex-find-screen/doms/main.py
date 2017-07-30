import numpy as np
import cv2
import util
from pyimagesearch import imutils

# image[0][0][BGR] - topleft
# image[0][1][BGR] - go right
# image[1][0][BGR] - go down
image = cv2.imread("../screen.jpg")
orig = image.copy()

# ratio = (height,width,channels)
# height = number of rows
# width = number of columns
ratio = image.shape[0] / 300.0
#image = cv2.resize(image, (0,0), fx=1/ratio, fy=1/ratio)

# NOTE: using a different function for resizing may
# result in failing to find the 4 corners
image = imutils.resize(image, height = 300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# help(cv2.findContours)
# Find contours
contours, hierarchy = cv2.findContours(
	edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Find the screen contour
screen_contour = None
for c in contours:
	# Approximate the contour
	# cv2.arcLength - calculates contour perimeter or curve length
	perimeter = cv2.arcLength(c, True)
	# cv2.approxPolyDP - approximate a curve or polygon
	# using the Douglas-Peucker algorithm
	approx = cv2.approxPolyDP(c, 0.02*perimeter, True)

	# Look for polygons that contain 4 corners
	# Problem: no 4 corner contour found
	# Use another contour for now
	#print len(approx)
	if len(approx) == 4:
		screen_contour = approx
		break

if screen_contour is None:
	print "Screen contour could not be found"
	exit()

# Switch from a 4x2x1 array to 4x2
# points = 4*(x,y) coordinates
points = screen_contour.reshape(4, 2)

# Find top/bottom, left/right corners
rect = util.get_corners(points)
rect *= ratio

# Find the width and height of the new image
max_width, max_height = util.get_width_height(rect)

# Create a destination array to
# map the screen to a top-down, "birds eye" view
dst = np.array([
	[0, 0],
	[max_width-1, 0],
	[max_width-1, max_height-1],
	[0, max_height-1]], dtype="float32")

# Calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(rect, dst)

# Warp the perspective to grab the screen
warp = cv2.warpPerspective(orig, M, (max_width, max_height))

cv2.imshow("image", image)
cv2.imshow("warp", imutils.resize(warp, height = 300))
cv2.waitKey(0)