import time
import logging 
import threading

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()

def faceDetect(img):
    dets = detector(img, 0)
    if not dets:
        return None
    face_location = max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))
    return face_location

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def keyPointsDetect(img, face_location):
    landmark_shape = predictor(img, face_location)
    keyPoints = []
    for i in range(68):
        pos = landmark_shape.part(i)
        keyPoints.append(np.array([pos.x, pos.y], dtype=np.float32))
    return keyPoints

def generateConstructor(keyPoints):
    def center(pos):
        return sum([keyPoints[i] for i in pos]) / len(pos)
    leftEyebrow = [18, 19, 20, 21]
    rightEyebrow = [22, 23, 24, 25]
    jaw = [6, 7, 8, 9, 10]
    nose = [29, 30]
    return center(leftEyebrow + rightEyebrow), center(jaw), center(nose)

def generateFeature(constructor):
    center_eyebrow, center_jaw, center_nose = constructor
    middleLine = center_eyebrow - center_jaw
    diagnal = center_eyebrow - center_nose
    h_rotate = np.cross(middleLine,diagnal) / np.linalg.norm(middleLine) ** 2
    v_rotate = middleLine @ diagnal / np.linalg.norm(middleLine) ** 2
    return np.array([h_rotate,v_rotate])

def extractFeature(img):
    face_location = faceDetect(img)
    if not face_location:
        return None
    keyPoints = keyPointsDetect(img, face_location)
    constructor = generateConstructor(keyPoints)
    rotate = generateFeature(constructor)
    return rotate 

feature = [0,0]
def captureLoop():
    global origin_feature
    global feature
    origin_feature = extractFeature(cv2.imread('standard.jpg'))
    feature = origin_feature - origin_feature
    cap = cv2.VideoCapture(0)
    logging.warning('Capture Start!')
    while True:
        ret, img = cap.read()
        new_feature = extractFeature(img)
        if new_feature is not None:
            feature = new_feature - origin_feature
        time.sleep(1/60)
    
def get_feature():
    return feature

thread = threading.Thread(target=captureLoop)
thread.setDaemon(True)
thread.start()
logging.warning('Capture Loop start...')

if __name__ == '__main__':
    while True:
        time.sleep(0.1)
        print(feature)
