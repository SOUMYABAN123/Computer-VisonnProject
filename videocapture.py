import cv2

cap = cv2.VideoCapture("C:/Users/HP/Downloads/video.mp4")
print("cap", cap)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, (700,450))
    cv2.imshow("frame", frame)
    k=cv2.waitKey(25)   # 0 means static - 25 means in n25 sec how many frames are getting desplayed
    if k== ord("q") & 0xFF:
        break
cap.release()
cv2.destroyAllWindows()