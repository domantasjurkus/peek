import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("../01.png", 0)
img2 = cv2.imread("../02.png", 0)
img1 = cv2.resize(img1, (1500,800))
img2 = cv2.resize(img2, (1500,800))
edges1 = cv2.Canny(img1, 20, 50)
edges2 = cv2.Canny(img2, 20, 50)

#merged = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)

# Horrible workaround (Dom, learn your numpy)
merged = []
for row in range(0, len(edges1)):
	new_row = []
	for i in range(0, len(edges1[0])):
		new_row.append(255 if edges1[row][i] and edges2[row][i] else 0)
	merged.append(new_row)

'''print len(edges1)
print len(edges1[0])
print len(edges2)
print len(edges2[0])'''

'''plt.subplot(121), plt.imshow(img1, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges1, cmap="gray")
plt.title("Edge Image"), plt.xticks([]), plt.yticks([])'''

plt.subplot(122), plt.imshow(merged, cmap="gray")
plt.title("Merged Image"), plt.xticks([]), plt.yticks([])
plt.show()
