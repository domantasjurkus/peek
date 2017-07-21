import numpy as np
import cv2
from matplotlib import pyplot as plt

hamster = cv2.imread("../hamster01.jpg", 0)
cv2.imshow('image', hamster)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('hamster_out.jpg', img)

'''plt.imshow(hamster, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.show()'''