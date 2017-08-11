import os
import cv2
from find_and_align import find_and_align
from find_difference import difference
from util import resize

'''
def parse_args():
	opts, args = getopt.getopt(sys.argv[1:], "", ["feature="])
	opts = dict(opts)
	#feature_name = opts.get("--feature", "surf")
	feature_name = opts.get("--feature", "sift")
	return feature_name
'''

def generate_background_images(src_directory="img/background/misaligned_fixed/", dst_directory="img/background/aligned_fixed/"):
	"""
	A set of background images are needed for the background subtraction algorithm
	used in difference.py
	"""
	control_path = "img/sample_control.jpg"
	img_control = cv2.imread(control_path)
	for filename in os.listdir(src_directory):
		if filename[-4:] != ".jpg":
			continue
		img = cv2.imread(src_directory+filename)
		img = resize(img)
		aligned = find_and_align.get_warped_image(img_control, img)

		save_path = dst_directory+filename
		cv2.imwrite(save_path, aligned);
		print "%s background image saved to %s" % (filename, save_path)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	control_path = "img/sample_control.jpg"
	query_path = "img/query1.jpg"
	img_control = cv2.imread(control_path)
	img_control = resize(img_control)

	img_query = cv2.imread(query_path)
	img_control = resize(img_control)

	img_query_aligned = find_and_align.get_warped_image(img_control, img_query)

	img_diff = difference.get_difference_image(img_control, img_query_aligned, True)
	quality_score = difference.get_quality_score(img_diff)

	print "Quality score: " + str(quality_score)
	cv2.imshow("control", img_control)
	cv2.imshow("query", img_query_aligned)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
	#generate_background_images()
