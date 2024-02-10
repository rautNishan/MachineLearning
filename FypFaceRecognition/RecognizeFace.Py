import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         print(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodesCurFrame = face_recognition.face_encodings(img)
#         encodeList.append(encodesCurFrame[0])
#     return encodeList

# encodeListKnown = findEncodings(images)
# print('Encoding Complete')

# cap = cv2.VideoCapture(0)

# FACE_DISTANCE_THRESHOLD = 0.60

# def similarity_score(face_distance):
#     scale = 0.6
#     return np.exp(-face_distance / scale)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)
#         score = similarity_score(faceDis[matchIndex])
#         print('Accuracy: ',score)
#         if matches[matchIndex] and score > FACE_DISTANCE_THRESHOLD:
#             name = classNames[matchIndex].upper()
            
#         else:
#             name = "Unknown"

#         y1, x2, y2, x1 = faceLoc
#         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)


class FaceRecognizer:
    def __init__(self, path='images'):
        self.path = path
        self.images = []
        self.classNames = []
        self.myList = os.listdir(path)
        self.encodeListKnown = []
        self.FACE_DISTANCE_THRESHOLD = 0.55
        self.load_images()

    def load_images(self):
        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = self.findEncodings(self.images)
        print('Encoding Complete')

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodesCurFrame = face_recognition.face_encodings(img)
            encodeList.append(encodesCurFrame[0])
        return encodeList

    def similarity_score(self, face_distance):
        scale = 0.6
        return np.exp(-face_distance / scale)

    def recognize(self):
        cap = cv2.VideoCapture(0)
        recognized_names = [] 
        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                score = self.similarity_score(faceDis[matchIndex])
                print('Accuracy: ',score)
                if matches[matchIndex] and score > self.FACE_DISTANCE_THRESHOLD:
                    name = self.classNames[matchIndex].upper()
                    print(name)
                    if not any(d['collegeId'] == name for d in recognized_names):
                        recognized_names.append({'collegeId':name, 'entryTime': datetime.now().strftime('%H:%M:%S')})# Store the current time
                else:
                    name = "Unknown"

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return recognized_names 