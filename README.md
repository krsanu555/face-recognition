# Face Recognition

In this project, we will implement face recognition system for identifying faces in images and also in live video.

### Project description

We have used [OpenCV](https://github.com/opencv) for face detection and [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) model for face recognition.FaceNet inputs a face image (or batch of m face images) as a tensor of shape (m,nC,nH,nW)=(m,3,96,96) and outputs a matrix of shape (m,128)that encodes each input face image into a 128-dimensional vector.

#### Face Encoding
We will encode the person faces and save into database(dictionary in our case).We are performing encodings for each person so that when system gets new images, it can compare with encodings of images stored in the database to identify who is that person. And this whole encoding process includes following steps:- <br />
Step 1. We will detect face from person image using Open CV Haar Cascade Face detector. <br />
Step 2. Crop those face from image using face location obtained in Step 1. <br />
Step 3. Resize face into (96,96,3) rgb channel. <br />
Step 4. Pass that face obtained in Step 3 to FaceNet Model which encodes input face into a 128-dimensional vector. <br />
Step 5. Add Person Name as key and encodings as value to the dictionary. <br />
Step 6. Repeat Step 1 to Step 5 for each images. <br />
Step 7. Finally serialize dictionary data to disk. <br />

Note:- For encodings, each image will contain only one face. <br />

#### Face Recognition
In this part we will perform face recognition in both images and live video.Here we will perform following steps:- <br />
Step 1. We will detect faces from image/live video using Open CV Haar Cascade Face detector. <br />
Step 2. Crop those faces from image using face location obtained in Step 1. <br />
Step 3. Resize those faces into (96,96,3) rgb channel. <br />
Step 4. Pass that faces obtained in Step 3 one by one to a method which will pass that face to the same FaceNet Model(used in face encoding process) to get 128-dimensional vector encoding and compare with face encodings stored in the disk. After this comparison, this method will return the name of the person based on the similarity or else return unknown(not found in database). <br />
Step 5. Then we will draw rectangle around each faces along with the label obtained in Step 4. <br />

Note:- For face recognition, each image can contain any number of faces. <br />


### Set up

1. Clone the repository
```
git clone https://github.com/krsanu555/face-recognition.git
```
2. Move inside the newly created repository
```
cd face-recognition
```
3. Create a virtual environment and install necessary dependecies
```
conda create --name my-env --file requirements.txt
```

### Run Application

1. Go to the root directory of the project
2. Activate virtual environment
```
conda activate my-env
```
3. Create folders(equal to the number of persons) in dataset directory with person name as folder name(it will be used as labels while recognition). <br />
4. Put images of each person to their corresponding folders which we have just created. <br />
5. Execute encode_faces.py python scripts using following command in command prompt. <br />
```
python encode_faces.py
```
This will create encodings for each face of each person and save these encodings by creating a file named ```encodings_faces.pickle``` in the root directory. <br /> 
6. Now put images(which you want to use in recognition) in testimages folder. <br />
7. Run following command to perform face recognition on images. <br />
```
python recognize_faces_image.py testimages/test.jpg
```
Here we are executing this recognize_faces_image.py python scripts to see the recognized faces in the input image(which is test.jpg in this case). And we are also saving that image with included face labels in the detectedimages folder.<br />

8. To perform face recognition in live video. Run following commands.<br />
```
python live_face_recognition.py
```
9. Deactivate environment
```
conda deactivate
```

Note:- For Steps 3,4 & 6 We have just put single image in each folder for better understanding.