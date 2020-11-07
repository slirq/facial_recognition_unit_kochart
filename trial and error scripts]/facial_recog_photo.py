import cv2
import numpy as np
import face_recognition
import os
import glob
path = 'ImageFolder'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def GetEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = GetEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 

#read from folder
newImg=str(glob.glob("*.jpg")[0])
img = cv2.imread(newImg,1)
imgS = cv2.resize(img,(0,0),None,0.25,0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
facesCurFrame = face_recognition.face_locations(imgS)
encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
    
    #check for lowest variation
    #faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
    
    #check highes probability
    faceDis = abs(1-face_recognition.face_distance(encodeListKnown,encodeFace))
    
    dictionary = dict(zip(classNames, faceDis))
    print(dictionary)
    #check for lowest variation
    #matchIndex = np.argmin(faceDis)
    
    #check for highest variation
    matchIndex = np.argmax(faceDis)
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print("ANSWER="+name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
#img = cv2.resize(img,(640,640))
cv2.imshow('Webcam',img)
cv2.waitKey(0)

