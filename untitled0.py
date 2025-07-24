import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("C://Users/HP/Downloads/contrastImage.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


hist = cv2.calcHist([image], [0], None, [256], [0,255])

cv2.imshow("Original Image", image)
plt.plot(hist)
plt.show()


image_hist = cv2.equalizeHist(image)
cv2.imshow("Histogram equalization", image_hist)
plt.plot(image_hist)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

"C:\Users\HP\Downloads\bvjegjgerj.jpeg"