from sklearn.neighbors import KNeighborsClassifier # Importing the KNN Classifier model to classify whos face it is.

import cv2
import pickle
import numpy as np
import os
import csv # Importing the csv module to save the data in the csv file for the sake of attendance recording.
import time # Importing the time module to get the time of attendance.
from datetime import datetime # Importing the datetime module to get the time of attendance.


from win32com.client import Dispatch # To make sure of speaking sound ( kindof.. voice assistant);


def speak(str1): # This function `speak()` is triggered when the user presses `o` to take the attendance. A sound will be played and the attendance will be recorded.
    
    speak=Dispatch(("SAPI.SpVoice")) # A package for voice sound.
    speak.Speak(str1) # Triggering the voice sound

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w: # Firstly, we gonna be loading the `names`(LABELS) and faces (FACES) from the pickle file.

    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f) # Similarly, this one is to load the faces from the pickle file.

print('Shape of Faces matrix --> ', FACES.shape)



# CREATING THE INSTANCE FOR K-NEAREST NEIGHBOUR CLASSIFIER AKA KNN CLASSIFIER AND HERE IN THIS CASE, WE TOOK K = 5.
knn=KNeighborsClassifier(n_neighbors=5) 
knn.fit(FACES, LABELS) # Fit the data (both FACES and LABELS (names)) to the KNN Classifier.



# LOADING THE BACKGROUND IMAGE.
imgBackground=cv2.imread("background.png") #


# CREATING THE LIST OF COLUMNS NAMES FOR THE ATTENDANCE SHEET.
COL_NAMES = ['NAME', 'TIME'] 

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        
        # THE ABOVE PART IS SAME AS IN `add_faces.py` file.
        
        
        
        
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1) # Here, we need to resize the image data  and since we're using machine learning algorithm, its necessary to flatten the data along with reshaping.
        
        
        
        
        output=knn.predict(resized_img) # Predicting the output of recognized face.



        ts=time.time() # Creating an instance for the time.



        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y") # Formatting the date.



        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S") # Formatting the time.


     # CHECKING THE EXISIENCE OF ATTENDANCE FILE FOR THE CURRENT DATE--> if it exists then it will loaded.
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")





        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)

        # The below code is for displaying the name of the recognized face.
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

        
        
        attendance=[str(output[0]), str(timestamp)] # STORING THE ATTENDANCE IN A LIST.
    imgBackground[162:162 + 480, 55:55 + 640] = frame

    
    cv2.imshow("Frame",imgBackground) # This is the main window where the video is displayed along with the background image.


    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("ATTENDANCE IS BEING TAKEN!!!..")
        time.sleep(0)
        if exist:

            # If it exists, then append the attendance to the existing csv file excluding the NAME.
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile) # Write the header of the csv file.
                writer.writerow(COL_NAMES) # Write the row for column names
                writer.writerow(attendance) # Write the row for attendance.
            csvfile.close()

    
    
    if k==ord('q') or k == ord('Q'):
        break
video.release()
cv2.destroyAllWindows()

