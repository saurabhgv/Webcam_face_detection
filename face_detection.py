import cv2 as cv

capture=cv.VideoCapture(0)
while True:
    isTrue,frame=capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_casced=cv.CascadeClassifier('haar_cascad.xml')
    face_rct=haar_casced.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=9)
    for (x,y,w,h) in face_rct:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.imshow('frame',frame)
    cv.waitKey(1)