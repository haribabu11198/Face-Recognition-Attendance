import face_recognition
import pathlib
import cv2
import numpy as np
import csv
import os
#import glob
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("C:\Users\harib\OneDrive\Desktop\Web\python\Face Recognition\images.jpg")
jobs_encoding =face_recognition.face_encodings(jobs_image)[0]

dinesh_image = face_recognition.load_image_file("C:\Users\harib\OneDrive\Desktop\Web\python\Face Recognition\images\dinesh.jpg")
dinesh_encoding =face_recognition.face_encodings(dinesh_image)[0]

ram_image = face_recognition.load_image_file("C:\Users\harib\OneDrive\Desktop\Web\python\Face Recognition\images\ram.jpg")
ram_encoding =face_recognition.face_encodings(ram_image)[0]

hari_image = face_recognition.load_image_file("C:\Users\harib\OneDrive\Desktop\Web\python\Face Recognition\images\hari.jpg")
hari_encoding =face_recognition.face_encodings(hari_image)[0]

prabhu_image = face_recognition.load_image_file("C:\Users\harib\OneDrive\Desktop\Web\python\Face Recognition\images\prabhu.jpg")
prabhu_encoding =face_recognition.face_encodings(prabhu_image)[0]

know_face_encoding = [
    jobs_encoding,
    dinesh_encoding,
    ram_encoding,
    hari_encoding,
    prabhu_encoding
]

know_faces_names = [
    "jobs",
    "dinesh",
    "ram",
    "hari",
    "prabhu"
]

friends = know_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.csv','w+',newline= "")
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding,face_encoding)
            name =""
            face_distance = face_recognition.face_distance(know_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = know_faces_names[best_match_index]
                
            face_names.append(name)
            if name in know_faces_names:
                if name in friends:
                    friends.remove(name)
                    print(friends)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.inshow("attendance system",frame)
    if cv2.waitkey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()