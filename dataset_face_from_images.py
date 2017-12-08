import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = 2

def extract_face(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    count = 0

    for imagePath in imagePaths:

        image_frame = cv2.imread(imagePath)

        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("Training Faces/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('frame', image_frame)

extract_face("Celebrities/Bill Gates")

