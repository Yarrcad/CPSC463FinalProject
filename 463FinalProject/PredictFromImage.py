import pickle
import cv2
import dlib
import math
import numpy as np

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("source\\shape_predictor_68_face_landmarks.dat")

with open('source/trained_svm_model', 'rb') as f:
    model = pickle.load(f)
image = cv2.imread('sorted_set/sadness/05_006_00000019.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe_image = clahe.apply(gray)

detections = detector(clahe_image, 1)
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
print("Anger:" + str(predictions[0][0]) + "\ncontempt:" + str(predictions[0][1]) + "\ndisgust:" + str(
    predictions[0][2]) + "\nfear:" + str(predictions[0][3]) + "\nhappiness:" + str(
    predictions[0][4]) + "\nneutral:" + str(predictions[0][5]) + "\nsadness:" + str(
    predictions[0][6]) + "\nsurprise:" + str(predictions[0][7]))
winner = 0
for i in range(0, 8):
    if predictions[0][i] > predictions[0][winner]:
        winner = i
print("\nThe WINNER is " + emotions[winner].upper() + " with a probability of " + str(predictions[0][winner]) + "!")
