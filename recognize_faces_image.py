##Usage python recognize_faces_image.py testimages/test.jpg

#importing necessary packages
import pickle
import cv2
from image_util import *
import sys
import os

cascPath = "haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# load the known faces and embeddings
print("[INFO] loading encodings...")
database = pickle.loads(open("encodings_faces.pickle", "rb").read())
print("[INFO] laoding Done!!...")

#Getting image path from command line
image_path=sys.argv[1]

#Loading FaceNet Model for Face Encodings
print("Loading Model Started(it will take few minutes)...")
model=get_facenet_model()
print("Loading Model Finished...")

# load the input image            
img = cv2.imread(image_path)
if (img is None):
    print("Can't open image file") 

#Converting image into Gray Scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Using Haar Cascade face detector for face location
faces =faceCascade.detectMultiScale(gray, 1.3,5, minSize=(100,100), flags = cv2.CASCADE_SCALE_IMAGE)
if (faces is None):
    print('Failed to detect face')                

facecnt = len(faces)
print("Detected faces: %d" % facecnt) 

#Getting extention and name of image
filename,ext= os.path.splitext(image_path)

#Looping over faces
i = 0     
for (x, y, w, h) in faces:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    
    #cropping face from image
    faceimg = img[ny:ny+nr, nx:nx+nr]
    #Resizing image to (96,96,3) so that it can be feed to FaceNet Model for face encodings
    resize_face = cv2.resize(faceimg, (96,96))
    
    #Identifying face from model
    name,_=recognize_face(resize_face, database,model)
    #Drawing box around the face
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)
    # Draw a label with a name above the face
    left=x
    top = y - 15 if y - 15 > 15 else y + 15
    cv2.putText(img,name, (left, top), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255),2)

# show the output image
cv2.imshow("Image", img)
#Writing image to the detectedimages folder
cv2.imwrite("detectedimages/"+filename.split('/')[-1]+ext, img)
cv2.waitKey(0)