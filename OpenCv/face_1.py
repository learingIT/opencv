import cv2


img = cv2.imread('image/5.jpg')

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(gray,1.1,8)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey()


