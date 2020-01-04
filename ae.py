import cv2
import numpy as np
import dlib
import pyautogui
from math import *
import os
import signal
from sklearn.linear_model import LogisticRegression


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\user1\\Desktop\\hitum\\facial-landmarks\\shape_predictor_68_face_landmarks.dat")

X_train=[]
y_train=[]
splitter=0

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
def speak():
    import pyttsx3
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    print (rate)                        #printing current voice rate
    engine.setProperty('rate', 125)
    volume = engine.getProperty('volume')
    print (volume)
    engine.setProperty('volume',1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say('We cannot detect you')
    engine.runAndWait()
    engine.stop()
def checker(eye):
    from sklearn.svm import SVC
    sv = LogisticRegression()
    sv.fit(X_train, y_train)
    return sv.predict(eye)
def trainer(x,y,z,w): 
    global X_train,y_train
    #print(X_train,y_train)
    x.extend(y)
    X_train.append(x)
    y_train.append(1)
    z.extend(w)
    X_train.append(z)
    y_train.append(0)
def clicker(x,y,z,w):
    global splitter
    splitter+=1
    if(splitter<=50):
        trainer(x,y,z,w)
        return
    x.extend(y)
    z.extend(w)
    if(checker([x])[0]==1):
        pyautogui.click()
    elif(checker([z])[0]==0):
        os.kill(os.getpid(),signal.SIGTERM)
    else:
        speak()
        pass
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        clicker(list(right_point),list(left_point),list(center_top),list(center_bottom))
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
