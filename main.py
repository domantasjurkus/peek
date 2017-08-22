import os, sys
import cv2
from src import align, difference, util

def parse_args():
	if len(sys.argv) < 3:
		control = "img/sample_control.jpg"
		query = "img/query1.jpg"
		show_images = 1
		print "Usage: python2 main.py <control-image> <query-image> [show-images=0]"
		print
		print "Running script with default samples:"
		print "Control: %s" % control
		print "Query: %s" % query
		print "Quality Score: ..."
	else:
		control = sys.argv[1]
		query = sys.argv[2]

	try:
		show_images = sys.argv[3]
	except:
		show_images = 0
	return (control, query, show_images)


def assess_quality(control_path, query_path, show_images=0):
	
	img_control = cv2.imread(control_path, 0)
	img_query = cv2.imread(query_path, 0)
	if img_control is None:
		print "error: cannot load control image %s - are you sure the path is right?" % control_path
		exit()
	if img_query is None:
		print "error: cannot load query image %s - are you sure the path is right?" % query_path
		exit()

	img_control = util.resize(img_control)
	img_control = util.resize(img_control)

	img_query_aligned = align.get_warped_image(img_control, img_query)

	img_diff = difference.get_difference_image(img_control, img_query_aligned, show_images)
	quality_score = difference.get_quality_score(img_diff)

	print str(quality_score)


if __name__ == "__main__":
	control_path, query_path, show_images = parse_args()
	assess_quality(control_path, query_path, show_images)
