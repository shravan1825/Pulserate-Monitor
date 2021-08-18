import cv2
import numpy as np
import dlib


hog_facedetector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor('shape_predictor_81_face_landmarks (1).dat')  #Using 81 facelandmarks

video = cv2.VideoCapture('jhbhhb.mp4')  #video file from which the face will be detected

while True:
    ret, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #to convert to greyscale
    faces = hog_facedetector(grayscale)  # DETECT FACES

    for face in faces:
        face_landmarks = dlib_facelandmark(grayscale, face)



        for n in range(0, 81):


            f = face_landmarks.part(71).x  #using face landmarks 21, 71, 22 for roi
            g = face_landmarks.part(71).y
            x = face_landmarks.part(21).x
            y = face_landmarks.part(21).y
            u = face_landmarks.part(22).x
            t = face_landmarks.part(22).y
            cv2.rectangle(frame, (f, g), (x, y), (255, 255, 255), 1)   #ROI
            x = int(((u-x)/2)+x)  #using three points in roi for accurate data
            y = int(((g-y)/2)+y)
            u = int(((u-x)/3)+x)
            t = int(((g-y)/3)+y)
            f = int(((u - x) / 4) + x)
            g = int(((t - y) / 4) + y)
            pulserate1 = frame[x, y, 1]  #considering only the green pixel color for pulse rate
            pulserate2 = frame[u, t, 1]
            pulserate3 = frame[f, g, 1]
            pulserate = int((pulserate3+pulserate2+pulserate1)/3)  #avg data of 3 points in the ROI for accuracy
            print(pulserate, 'is the pulse rate')



        cv2.imshow('Face Landmarks', frame)

    g = cv2.waitKey(30) & 0xff
    if g == 115:   #press s to stop
        break

video.release()   # to realese video capture object
cv2.destroyAllWindows()
