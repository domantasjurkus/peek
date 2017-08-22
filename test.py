import os
from main import assess_quality

def get_images_from_dir(directory):
	images = []
	for filename in os.listdir(directory):
		if filename[-4:] != ".jpg":
			continue
		images.append(directory+filename)
	return images


def test_directory(directory="img/test/arbitrary_lighting/"):
	control_image = directory + "control.jpg"
	good_images = get_images_from_dir(directory + "good/")
	bad_images = get_images_from_dir(directory + "bad/")

	for g in good_images:
		print "python main.py %s %s   " % (control_image, g),
		assess_quality(control_image, g)

	for b in bad_images:
		print "python main.py %s %s   " % (control_image, b),
		assess_quality(control_image, b)
	print


if __name__ == "__main__":
	test_directory()
	test_directory("img/test/controlled_lighting/")
