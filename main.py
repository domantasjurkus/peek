import os, sys
import cv2
from find_and_align import find_and_align
from find_difference import difference
from util import resize

def parse_args():
	if len(sys.argv) < 4:
		control = "img/sample_control.jpg"
		query = "img/query1.jpg"
		show_images = 0
		print "Usage: python2 main.py <control-image> <query-image> [show-images=0]"
		print
		print "Running script with default samples:"
		print "Control: %s" % control
		print "Query: %s" % query
		print "Quality Score: ..."
	else:
		control = sys.argv[1]
		query = sys.argv[2]
		show_images = sys.argv[3]
	return (control, query, show_images)

"""def generate_background_images(src_directory="img/background/misaligned_fixed/", dst_directory="img/background/aligned_fixed/"):
	# A set of background images are needed for the background subtraction algorithm
	# used in difference.py
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
	cv2.destroyAllWindows()"""

def main(control_path, query_path, show_images=0):
	
	img_control = cv2.imread(control_path, 0)
	img_control = resize(img_control)

	img_query = cv2.imread(query_path, 0)
	img_control = resize(img_control)

	img_query_aligned = find_and_align.get_warped_image(img_control, img_query)

	img_diff = difference.get_difference_image(img_control, img_query_aligned, show_images)
	quality_score = difference.get_quality_score(img_diff)

	print str(quality_score)

if __name__ == "__main__":
	control_path, query_path, show_images = parse_args()
	main(control_path, query_path, show_images)
	#generate_background_images()
