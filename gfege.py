import cv2
import matplotlib.pyplot as plt

# Load and preprocess the image
image = cv2.imread("C://Users/HP/Downloads/demoimage123.jpeg")  # Replace with your image path. Resizes the image to 64x128 pixels.
image = cv2.resize(image, (64, 128))  # Resize to standard HOG size.This is a standard size used in HOG (Histogram of Oriented Gradients) feature extraction. It simplifies processing and ensures consistency.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradients using Sobel operator
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # Horizontal gradient, Detects vertical edges. CV_32F ensures precision, and ksize=1 uses a small kernel for fine details.
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # Vertical gradient,  Detects horizontal edges. Together with gx, it gives full edge information.

# Compute magnitude and angle of gradients
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)  #Converts the gradients from Cartesian (gx, gy) to polar coordinates.

# Normalize magnitude for visualization
magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  #Makes the gradient strength visually interpretable in grayscale images.

# Display results
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gradient Magnitude")
plt.imshow(magnitude_norm, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Gradient Orientation")
plt.imshow(angle, cmap='hsv')  # HSV colormap for angle visualization, HSV colormap makes it easy to distinguish different edge directions.
plt.axis('off')

plt.tight_layout()
plt.show()
