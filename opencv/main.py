import cv2
from find_and_align import find_and_align
'''(Intended) Usage:
$ python main.py control_image query_image

'''

'''def parse_args_to_be_used():
	# TODO
	opts, args = getopt.getopt(sys.argv[1:], "", ["feature="])
	opts = dict(opts)
	#feature_name = opts.get("--feature", "surf")
	feature_name = opts.get("--feature", "sift")
	return feature_name'''


def main():
	# Import control and query images
	control_path = "img/control.jpg"
	query_path = "img/damaged_pen.jpg"
	img_control = cv2.imread(control_path, 0)
	img_query = cv2.imread(query_path, 0)

	# Align query image
	img_query = find_and_align.get_warped_image(img_control, img_query)
	
	cv2.imshow("control", img_control)
	cv2.imshow("query", img_query)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	# Compare the images, produce numeric outputs


if __name__ == "__main__":
	main()