import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("C:/Users/HP/Downloads/video.mp4")

# Parameters
min_width_rect = 80
min_height_rect = 80
count_line_position = 550
offset = 6  # Allowable error for detecting line crossing

# Initialize background subtractor
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    """Returns center point of the bounding box"""
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

# Detection and counter
detect = []
vehicle_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Background subtraction and morphological operations
    img_sub = algo.apply(blur)
    dilated = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_valid = (w >= min_width_rect) and (h >= min_height_rect)
        if not is_valid:
            continue

        # Draw bounding boxes and centers
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    # Check if object crosses the line
    for (x, y) in detect:
        if count_line_position - offset < y < count_line_position + offset:
            vehicle_counter += 1
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove((x, y))
            print("Vehicle Counter:", vehicle_counter)

    # Display count on screen
    cv2.putText(frame, "Vehicle Count: " + str(vehicle_counter), (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Resize and display frame
    frame = cv2.resize(frame, (1000, 700))
    cv2.imshow("Vehicle Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
