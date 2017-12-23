import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
AREA = 100

c = 0
cor = [0]
TRESH_HOLD = 20
display=""

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while (1):
    flag1=False
    flag2=False
    flag3=False
    flag4=False

    ret, frame = cap.read()
    orignal = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    thresh = fgbg.apply(gray)

    kernel_square = np.ones((3, 3), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    thresh = cv2.dilate(thresh, kernel_ellipse, iterations=1)
    thresh = cv2.erode(thresh, kernel_square, iterations=1)
    filtered = cv2.medianBlur(thresh, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)

    cv2.imshow('Motion', thresh)

    im2, contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = [cv2.moments(cnt) for cnt in contours0]

    cv2.line(frame,(0,100),(1000,100),(255,0,0),3,cv2.FONT_HERSHEY_SIMPLEX)

    cv2.line(frame,(100,0),(100,1000),(255,0,0),3,cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(frame,(325,0),(325,1000),(255,0,0),3,cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(frame,(550,0),(550,1000),(255,0,0),3,cv2.FONT_HERSHEY_SIMPLEX)

    for cnt in contours0:

        if cv2.moments(cnt)['m00'] > AREA:
            rect = cv2.minAreaRect(cnt)
            x, y = rect[0]
            w, h = rect[1]
            if x<100:flag1=True
            if x>550:flag4=True
            if x<325 and y<100:flag2=True
            if x>325 and y<100:flag3=True
            cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 0, 255), 2)

    cv2.imshow("Reference", frame)

    # cv2.putText(orignal,"Test",(500,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),3)

    if flag1 and flag4:
        display="Wide"
    elif flag2 and flag3:
        display="Six"
    elif (not flag3) and flag2:
        display="Out"

    if display=="Wide":
        cv2.putText(orignal, "Wide", (500, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),3)
    elif display=="Six":
        cv2.putText(orignal, "Six", (500, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),3)
    elif display=="Out":
        cv2.putText(orignal, "Out", (500, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255),3)

    cv2.imshow("Orignal", orignal)
    out.write(orignal)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


out.release()
cap.release()
cv2.destroyAllWindows()
