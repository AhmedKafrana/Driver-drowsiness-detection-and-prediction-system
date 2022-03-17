# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 21:50:38 2022

@author: hmoha
"""
import numpy as np
import cv2

import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import tensorflow as tf
###############################################################
##                    Video Input Capture                    ##
###############################################################

## Use an offline video
#vid = cv2.VideoCapture('red-ball.webm')
vid = cv2.VideoCapture('Driver_Drowsiness.mp4')

## Use real-time video capturing
#cameraID = 0 # 0 => Laptop's webcam ID
#vid = cv2.VideoCapture(cameraID)
model = tf.keras.models.load_model("Driver_Drowsiness_Detection.h5")

# cascPathface = os.path.dirname(
#     cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# cascPatheyes = os.path.dirname(
#     cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

# face_cascade = cv2.CascadeClassifier('driver_drowsiness_detection/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('driver_drowsiness_detection/haarcascade_eye.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('driver_drowsiness_detection/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)
# faceCascade = cv2.CascadeClassifier(cascPathface)
# eyeCascade = cv2.CascadeClassifier(cascPatheyes)
def full_face_detection_pipeline(input_image_path):
    test_image = cv2.imread(input_image_path)
    test_image = imutils.resize(test_image, width=800)
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    rects = detector(test_image_gray, 2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(test_image[y:y+h, x:x+w], width=256)
        faceAligned = fa.align(test_image, test_image_gray, rect)
        faceAligned_gray = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        plt.imshow(faceAligned_gray)
        plt.axis('off')
        plt.title('Aligned Face')
        plt.show()
        eyes = eye_cascade.detectMultiScale(test_image, 1.1, 4)
        predictions = []
        for (ex, ey, ew, eh) in eyes:
            eye = faceAligned[ey:ey+eh, ex:ex+ew]
#             cv2.rectangle(test_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 8)
            eye = cv2.resize(eye, (32, 32))
            eye = np.array(eye)
            eye = np.expand_dims(eye, axis=0)
            ypred = model.predict(eye)
            ypred = np.argmax(ypred[0], axis=0)
            predictions.append(ypred)
        if all(i==0 for i in predictions):
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 8)
            cv2.putText(test_image, 'Sleeping', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 8)
            cv2.putText(test_image, 'Not Sleeping', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    output_path = 'driver_drowsiness_detection/test_image_prediction.jpg'
    #cv2.imwrite(output_path, test_image) 
    return output_path

while True:
    ## Capture one frame from the video capture
    _, image = vid.read()
    figure = plt.figure(figsize=(5, 5))
    
    predicted_image = cv2.imread(full_face_detection_pipeline(image))
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    plt.imshow(predicted_image)
    plt.axis('off')
    plt.show()
    ## Break key
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
vid.release()
cv2.destroyAllWindows()