""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

red = (0,0,255)
white = (0,0,0)
blue = (255,0,0)
black = (255,255,255)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    kernel = np.ones((21,21), 'uint8')

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)

        # mouse
        cv2.ellipse(frame,(int(x+w*0.5),int(y+h*0.65)),(100,50),0,0,180,red,-1)

        # Nose
        cv2.line(frame, (int(x+w*0.47), int(y+h*0.55)), (int(x+w*0.53), int(y+h*0.55)), black, 10)
        cv2.line(frame, (int(x+w*0.5), int(y+h*0.55)), (int(x+w*0.5), int(y+h*0.4)), black, 10)

        # eyes
        cv2.circle(frame, (int(x+w*0.7), int(y+h*0.35)), 25, black, -1)
        cv2.circle(frame, (int(x+w*0.3), int(y+h*0.35)), 25, black, -1)
        cv2.circle(frame, (int(x+w*0.3), int(y+h*0.35)), 20, blue, -1)
        cv2.circle(frame, (int(x+w*0.7), int(y+h*0.35)), 20, blue, -1)
        cv2.circle(frame, (int(x+w*0.7), int(y+h*0.40)), 15, white, -1)
        cv2.circle(frame, (int(x+w*0.3), int(y+h*0.40)), 15, white, -1)

        # comments
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'You looks good!',(int(x+w*0.2),int(y+h*0.9)), font, 1,black,2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
