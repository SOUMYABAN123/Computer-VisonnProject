# Chagning a user input image to grayscale image
import cv2
path = input("enter the path and name of the image==")
print("you enter this===", path)

img1 = cv2.imread(path, 0) 
img1 = cv2.resize(img1,(1280, 700))
print(img1)
cv2.imshow("Converted image==", img1)
k=cv2.waitKey(0) # default value 0 - controling the time
if k==ord("s"):
    cv2.imwrite("Downloads:\\output.jpg", img1)
else:
    cv2.destroyAllWindows()