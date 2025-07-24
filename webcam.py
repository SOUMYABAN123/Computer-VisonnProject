import cv2

cap = cv2.VideoCapture(0)
print("cap", cap)

while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (700,450))
        #gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)
        #cv2.imshow("frame", gray)
        k=cv2.waitKey(25)   # 0 means static - 25 means in n25 sec how many frames are getting desplayed
        if k== ord("q") & 0xFF:
            break
cap.release()
cv2.destroyAllWindows()