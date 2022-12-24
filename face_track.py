import dlib
import cv2
import numpy as np



detector = dlib.get_frontal_face_detector()

def faceDetect(img):
    dets = detector(img, 0)
    if not dets:
        return None
    faceLoc = max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))
    return faceLoc

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def keyPointsDetect(img, faceLoc):
    landmark_shape = predictor(img, faceLoc)
    keyPoints = []
    for i in range(68):
        pos = landmark_shape.part(i)
        keyPoints.append(np.array([pos.x, pos.y], dtype=np.float32))
    for i, (px, py) in enumerate(keyPoints):
        cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
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

def draw(h_rotate, v_rotate):
    img = np.ones([512, 512], dtype=np.float32)
    face_len = 200
    center = 256, 256
    left_eye = int(220 - h_rotate * face_len), int(249 + v_rotate * face_len)
    right_eye = int(292 - h_rotate * face_len), int(249 + v_rotate * face_len)
    mouse = int(256 - h_rotate * face_len / 2), int(310 + v_rotate * face_len / 2)
    cv2.circle(img, center, 100 , 0, 1)
    cv2.circle(img, left_eye, 15, 0, 1)
    cv2.circle(img, right_eye, 15, 0, 1)
    cv2.circle(img, mouse, 5, 0, 1)
    return img

def extractFeature(img):
    face_location = faceDetect(img)
    if not face_location:
        cv2.imshow('self', img)
        cv2.waitKey(1)
        return None
    
    keyPoints = keyPointsDetect(img, face_location)
    constructor = generateConstructor(keyPoints)
    for i, (px, py) in enumerate(constructor):
        cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    rotate = generateFeature(constructor)
    cv2.putText(img, '%.3f' % rotate[0],
                (int(constructor[-1][0]), int(constructor[-1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    cv2.imshow('self', img)
    return rotate

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    origin_feature = extractFeature(cv2.imread('standard.jpg'))
    feature = origin_feature - origin_feature
    while True:
        ret, img = cap.read()
        img = cv2.flip(img,1)
        new_feature = extractFeature(img)
        if new_feature is not None:
            feature = new_feature - origin_feature
        h_rotate, v_rotate = feature
        cv2.imshow('Vtuber', draw(h_rotate, v_rotate))
        key = cv2.waitKey(1)
        if key == ord('q'):
            quit()