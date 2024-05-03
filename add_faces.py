import cv2 # To capture the camera data.
import pickle # To save the data in the form of pickle file.
import numpy as np # To convert the faces_data to numpy array.
import os # This is to create and check and save the files.--> File handling..


video=cv2.VideoCapture(0)  # Represents that you're opening your inbuilt camera.
# Suppose if you want to use the external web-camera, set it to '1'. i.e, '.VideoCapture(1)'


# This is the algorithm which we gonna be using to detect our faces in the video--> which xml file we're gonna be using is "haarcascade Classifier"
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[] # This list is gonna be used to store the faces data.

i=0

name=input("Enter Your Name: ") # This is gonna take the name of the person.

while True: # Creating an infinite loop

    ret,frame=video.read() # THis is gonna read the camera data that takes mainly 2 parameters, ret(boolean value either 1 or 0) and frame

    
    # This is neccessary to be added. Because, OPENCV interprets image data using BGR (Blue, Green, Red) instead of RGB (Red, Green, Blue).
    # Reason : because, historically opencv was developed in C++ and it was developed in BGR format.
    # So, to convert the image data from BGR to RGB, we're gonna use the following function.
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces=facedetect.detectMultiScale(gray, 1.3 ,5)# This is gonna detect our face with grayscale frame (defined under `gray` variable. followed by giving the threshold value for detection frame(x,y)-->(1.3,5).)



    #The below one, is the loop to get the dimensions of x and y and width and height of the face.
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :] # Cropping the image with the help of the coordinates of the face.


        resized_img=cv2.resize(crop_img, (50,50)) # Resizing the image with the help of the coordinates of the face.

        if len(faces_data)<=100 and i%10==0: # we need to stop adding images. Also we will be taking 10 frames so hence we are taking the modulus by 10.
            faces_data.append(resized_img) # Whatever the new faces detected, it will be added to `faces_data` list.
        i=i+1 # Increment the value of i --> increment number of frames added.



        # This is basically to tell how many pictures u can add. It takes frame, length of `faces_data`, coordinates,font face, font size, color, thickness.
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (239,167,239), 1)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (239,167,239), 1) # Rectangle frame followed by giving the coordinate values along with width and height, followed by (50,50,225)--> RED COLOR which the frame border color and 1 represents the thickness
    cv2.imshow("Frame",frame) 

    k=cv2.waitKey(1) # Defining a binding function, to wait until the user presses a key.


    if k==ord('q') or k == ord('Q') or len(faces_data) == 100:  # Suppose if the user input pressed is 'q' or 'Q' then the loop will break(Exits).
        break

video.release()
cv2.destroyAllWindows()



faces_data=np.asarray(faces_data) # Converting the list to array.
faces_data=faces_data.reshape(100, -1) # This is basically to reshape the array.



# NOW WE'RE GOING TO SAVE THE "NAME OF THE PERSON" IN THE FORM OF PICKLE FILE.

if 'names.pkl' not in os.listdir('data/'): # This is basically to check if the file is present or not.
    names=[name]*100
    with open('data/names.pkl', 'wb') as f: # This will open the file in write mode.(In terms of binary format.)
        pickle.dump(names, f) # Here we are dumping the data into the pickle file.
else:
    with open('data/names.pkl', 'rb') as f: # if `names.pkl` file is present then it will open or we gonna load  the file in read mode.
       
        names=pickle.load(f) # Here, we are tryna load the data from the pickle file.

    names=names+[name]*100 # Say suppose, the name "preethika" is already available in the pickle file, and u wanna add "priyal" then we just have to concatenate with other names.


    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f) # This is the same step as followed, inorder to dump the data into the pickle file.






# SIMILAR TO ABOVE, WE'RE GOING TO DUMP THE `faces_data` (Face datas) IN THE FORM OF PICKLE FILE.

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f) # Dump the newly added face datas.
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f) # load the detected face datas.
    faces=np.append(faces, faces_data, axis=0) # Appending the loaded face datas.
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f) # Dump the detected/loaded face datas.