import cv2
import numpy as np
import face_recognition
import os

path = 'imagesAttn'

images = []
classNames = []
# read all the images automatically
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        print(img)
        update_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(update_img)[0]
        encodeList.append(encode)
    return encodeList


# def markAttendance(name):
#     with open('attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtstring = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name}, {dtstring}')



encodeListOfKnownFaces = findEncoding(images)
print(len(encodeListOfKnownFaces))
print(images)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(frame)
    eframe = face_recognition.face_encodings(frame,faceLoc)
    cv2.imshow('Frame', frame)

    for encodeFace, faceLocations in zip(eframe,faceLoc):
        matches = face_recognition.compare_faces(encodeListOfKnownFaces, encodeFace)
        distance = face_recognition.face_distance(encodeListOfKnownFaces, encodeFace)

        print(distance)
        matchIndex = np.argmin(distance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocations
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2 )
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)

            temp = name + ' ' + str(round((1 - distance[matchIndex])*100, 2)) + '%'

            cv2.putText(frame, temp, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2 )
            

    cv2.imshow('Face', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

