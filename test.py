import os
from main import main

def get_images_from_dir(directory):
	images = []
	for filename in os.listdir(directory):
		if filename[-4:] != ".jpg":
			continue
		images.append(directory+filename)
	return images

def testmain():
	control_image = "test/control.jpg"
	good_images = get_images_from_dir("test/good/")
	bad_images = get_images_from_dir("test/bad/")

	for g in good_images:
		print "Control against %s:" % g,
		main(control_image, g)

	for b in bad_images:
		print "Control against %s:" % b,
		main(control_image, b)

if __name__ == "__main__":
	testmain()