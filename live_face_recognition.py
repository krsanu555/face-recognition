##Usage python live_face_recognition.py

# import the necessary packages
import pickle
import cv2
from image_util import *
import sys
import os
import time


# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load the known faces and embeddings
print("[INFO] loading encodings...")
database = pickle.loads(open("encodings_faces.pickle", "rb").read())
print("[INFO] laoding Done!!...")

#Loading FaceNet Model for Face Encodings
print("Loading Model Started(it will take few minutes)...")
model=get_facenet_model()
print("Loading Model Finished...")

cap = cv2.VideoCapture(0)
while(True):
    #Capture frame by frame
    ret, frame = cap.read()
    #Converting frame into Gray Scale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Using Haar Cascade face detector for face location
    faces = faceCascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5,minSize=(100, 100),flags = cv2.CASCADE_SCALE_IMAGE)
    
    print("Found {0} faces!".format(len(faces)))
    
    names = []
    #looping over faces
    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        
        #cropping face from image
        faceimg = frame[ny:ny+nr, nx:nx+nr]
        #Resizing image to (96,96,3) so that it can be feed to FaceNet Model for face encodings
        resize_face = cv2.resize(faceimg, (96,96))
        
        #Identifying face through model
        name,dist=recognize_face(resize_face, database,model) 
        #Appending identified face to names array
        names.append(name)
        
    # Draw a rectangle around the faces and putting name over face
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        y = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
