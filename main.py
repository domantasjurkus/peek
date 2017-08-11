import cv2
from find_and_align import find_and_align
from find_difference import difference

'''
def parse_args():
	opts, args = getopt.getopt(sys.argv[1:], "", ["feature="])
	opts = dict(opts)
	#feature_name = opts.get("--feature", "surf")
	feature_name = opts.get("--feature", "sift")
	return feature_name
'''

def main():
	control_path = "img/sample_control.jpg"
	query_path = "img/sample_query.jpg"
	img_control = cv2.imread(control_path, 0)
	img_query = cv2.imread(query_path, 0)

	img_query_aligned = find_and_align.get_warped_image(img_control, img_query)

	img_diff = difference.get_difference_image(img_control, img_query_aligned)
	quality_score = difference.get_quality_score(img_diff)

	print "Quality score: " + str(quality_score)
	cv2.imshow("control", img_control)
	cv2.imshow("query", img_query_aligned)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()