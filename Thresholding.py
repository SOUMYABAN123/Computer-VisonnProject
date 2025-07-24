import cv2
import numpy as np

def global_adaptive_threshold(image, epsilon=1.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = np.mean(gray)
    delta = float('inf')

    while delta > epsilon:
        lower = gray[gray <= T]
        upper = gray[gray > T]

        if lower.size == 0 or upper.size == 0:
            break

        mu1 = np.mean(lower)
        mu2 = np.mean(upper)
        new_T = (mu1 + mu2) / 2
        delta = abs(T - new_T)
        T = new_T

    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return binary

def local_adaptive_threshold(image, window_size=(15, 15)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Ww, Hw = window_size
    pad_w, pad_h = Ww // 2, Hw // 2

    # Pad the image
    padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    result = np.zeros_like(gray)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            window = padded[y:y+Hw, x:x+Ww]
            local_mean = np.mean(window)
            result[y, x] = 255 if gray[y, x] > local_mean else 0

    return result

if __name__ == "__main__":
    image = cv2.imread("C://Users/HP/Downloads/avengers.jpeg")  # Replace with your image path

    global_result = global_adaptive_threshold(image, epsilon=1.0)
    local_result = local_adaptive_threshold(image, window_size=(15, 15))

    cv2.imshow("Global Adaptive Thresholding", global_result)
    cv2.imshow("Local Adaptive Thresholding", local_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("global_result.jpg", global_result)
    cv2.imwrite("local_result.jpg", local_result)
