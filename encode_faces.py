##Usage python encode_faces.py

# import the necessary packages
import pickle
import cv2
import os
from image_util import *


cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


#Getting total number of images in dataset directory
total_images=len([x for name in os.listdir('dataset') for x in os.listdir(os.path.join('dataset', name)) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])

#Loading FaceNet Model for Face Encodings
print("Loading Model Started(it will take few minutes)...")
model=get_facenet_model()
print("Loading Model Finished...")

# initialize the dictionary for stroring encodings and person names
database={}


total=1

#Iterating over the dataset directory to get images
for (rootDir, dirNames, filenames) in os.walk('dataset'):
    
        # loop over the filenames in the current directory
        for filename in filenames:            
            print("Processing image {}/{}".format(total,total_images))
            #Path of current image
            image_path=os.path.join(rootDir, filename)
            #Getting name of person(i.e parent folder name containing images)
            name=image_path.split('\\')[1] 
            #Image extension
            _ , ext= os.path.splitext(image_path)
            #Reading image
            img = cv2.imread(image_path)            
            if (img is None):
                print("Can't open image file")                
            #Converting image into Gray Scale image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #Using Haar Cascade face detector for face location
            faces =faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100), flags = cv2.CASCADE_SCALE_IMAGE)
            if (faces is None):
                print('Failed to detect face')                

            facecnt = len(faces)
            print("Detected faces: %d" % facecnt)            
            
            i = 0      
            #looping over faces
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
                
                i += 1
                facename=str(total)+str(i) 
                
               #Uncomment this below (line 78-85) if you want to save cropped face to cropped_faces folder so that you can see and tune
               #parameter of detectMultiScale method(Haar cascade Face Detector) accordingly for better face detection

#                 if not os.path.exists("cropped_faces/"+name):
#                     os.makedirs("cropped_faces/"+name) #                 
#                 imagepath="cropped_faces/{0}/{1}".format(name,facename+ext)               
                
#                 try:
#                     cv2.imwrite(imagepath, resize_face)
#                 except:                   
#                     print("Failed to save cropped face")

                #Getting face encodings
                encoding =get_image_encoding(resize_face,model)
                #Storing Encodings into database dictionary
                database[name+"_"+facename] =encoding 
                    
            total+=1 
            
# dump the facial encodings + names to disk
print("Serializing encodings...")
f = open("encodings_faces.pickle", "wb")
f.write(pickle.dumps(database))
f.close()
print("Serialization Done!...")