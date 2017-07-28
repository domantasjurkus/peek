import cv2
import numpy as np

im1 =  cv2.imread("../../images/coaster/main.jpg");
im2 =  cv2.imread("../../images/coaster/rotated/03.jpg");
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
sz = im1.shape
warp_mode = cv2.MOTION_EUCLIDEAN

if warp_mode == cv2.MOTION_HOMOGRAPHY :
	warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
	warp_matrix = np.eye(2, 3, dtype=np.float32)
 
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
 
if warp_mode == cv2.MOTION_HOMOGRAPHY :
	# Use warpPerspective for Homography
	im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
	# Use warpAffine for Translation, Euclidean and Affine
	im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
cv2.imshow("Image 1", im1)
cv2.imshow("Image 2", im2)
cv2.imshow("Aligned Image 2", im2_aligned)
cv2.waitKey(0)