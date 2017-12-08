import cv2

# Import numpy for matrices calculations
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:

        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        Id = recognizer.predict(gray[y:y+h,x:x+w])

        name = "Unknown"

        if(Id[0] == 1):
            name = "Vasu"
        else:
            name = "Unknown"

        print(Id)

        confidence = str(Id[1])[0:2] + '%'

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(name) + " " + confidence, (x,y-40), font, 1, (255,255,255), 1)

    cv2.imshow('im',im)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()