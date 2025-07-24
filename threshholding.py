import cv2
import numpy as np

def global_adaptive_threshold(image, epsilon=1.0):     #Defines a function that finds a global adaptive threshold. 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #epsilon: The stopping criteria for threshold convergence.
    T = np.mean(gray)
    delta = float('inf')                               #Sets delta to infinity initially to enter the while loop.

    while delta > epsilon:                              #Repeats threshold updating until changes between iterations are small (i.e. convergence).
        lower = gray[gray <= T]                         #Separates pixel intensities into two groups: those below and above the threshold.
        upper = gray[gray > T]

        if lower.size == 0 or upper.size == 0:          #Prevents divide-by-zero or empty slice error.
            break

        mu1 = np.mean(lower)                            #Computes new threshold as average of means of both regions.
        mu2 = np.mean(upper)
        new_T = (mu1 + mu2) / 2
        delta = abs(T - new_T)
        T = new_T

    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)    #Applies the final threshold to create a binary image.
    return binary

def local_adaptive_threshold(image, window_size=(51, 51)):       #Defines function using a moving window for thresholding.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               #Breaks window size into width and height halves for padding.
    Ww, Hw = window_size
    pad_w, pad_h = Ww // 2, Hw // 2

    # Pad the image
    padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)   #Reflectively pads the image to avoid boundary issues during windowing python
    result = np.zeros_like(gray)

    for y in range(gray.shape[0]):                                                  #Loops over every pixel in the image.
        for x in range(gray.shape[1]):                                               #Extracts a window around the pixel, computes its mean, and thresholds accordingly.
            window = padded[y:y+Hw, x:x+Ww]
            local_mean = np.mean(window)
            result[y, x] = 255 if gray[y, x] > local_mean else 0

    return result

if __name__ == "__main__":
    image = cv2.imread("C://Users/HP/Downloads/demoimage.jpg")  

    global_result = global_adaptive_threshold(image, epsilon=1.0)
    local_result = local_adaptive_threshold(image, window_size=(51, 51))

    cv2.imshow("Global Adaptive Thresholding", global_result)
    cv2.imshow("Local Adaptive Thresholding", local_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("global_result.jpg", global_result)
    cv2.imwrite("local_result.jpg", local_result)
