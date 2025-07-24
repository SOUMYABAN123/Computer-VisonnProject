import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("C://Users/HP/Downloads/bvjegjgerj.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()    #Creates a HOG descriptor object. HOG is used to extract features based on edge directions and gradients.

# Compute HOG features
hog_features = hog.compute(gray)    #Computes the HOG feature vector for the grayscale image.
                                    #Prints the shape of the feature vector (useful for understanding the dimensionality of the data).

# Print feature vector shape
print("HOG feature vector shape:", hog_features.shape)   

# Optional: Use HOG for pedestrian detection
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())    #Sets a pre-trained SVM detector for pedestrian detection.
                                                                    #detectMultiScale scans the image at multiple scales to detect people.
                                                                    #winStride=(8,8) controls the step size of the sliding window.
boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))

# Draw bounding boxes
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)       #Draws green rectangles around detected pedestrians.

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Pedestrians")
plt.axis('off')
plt.show()











