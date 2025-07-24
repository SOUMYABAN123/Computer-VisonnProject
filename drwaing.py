import cv2
import numpy as np

# Original Image - 3D Array
img1 = cv2.imread("C://Users/HP/Downloads/avengers.jpeg", 1) # Original = 1
img1 = cv2.resize(img1,(1280, 700))
print(img1)

img1 = cv2.line(img1,(0,0),(200,200),(154,92,424),5)
img1 = cv2.arrowedLine(img1,(0,0),(200,200),(154,92,424),5)
cv2.imshow("original", img1)
cv2.waitKey(0) # default value 0 - controling the time
cv2.destroyAllWindows()
