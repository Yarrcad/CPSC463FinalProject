import pickle
import cv2
import dlib
import math
import sys
import numpy as np
import logging as log
import datetime as dt
from time import sleep

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("source\\shape_predictor_68_face_landmarks.dat")
cascPath = "cv2_cascades\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


def imagePredict():
    with open('source/trained_svm_model', 'rb') as f:
        model = pickle.load(f)
    image = cv2.imread('face.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1)
    if len(detections) == 0:
        return "NO FACE"
    for k, d in enumerate(detections):
        shape = predictor(clahe_image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = [[]]
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised[0].append(w)
            landmarks_vectorised[0].append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised[0].append(dist)
            landmarks_vectorised[0].append((math.atan2(y, x) * 360) / (2 * math.pi))
    predictions = model.predict_proba(landmarks_vectorised)
    winner = 0
    for i in range(0, 8):
        if predictions[0][i] > predictions[0][winner]:
            winner = i
    return emotions[winner] + " " + str(predictions[0][winner])


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    cv2.imwrite('face.jpg', frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, imagePredict(), (0, 130), font, 1, (200, 255, 155))

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    for k, d in enumerate(detections):
        shape = predictor(clahe_image, d)
        for i in range(1, 68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255),
                       thickness=2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
