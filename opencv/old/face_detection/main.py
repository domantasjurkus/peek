import numpy as np
import cv2

# Watch out for file path imports
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

try:
    img = cv2.imread('../../images/people.jpg')
    crimson = cv2.imread('../../images/smileys/crimson.jpg')
    len(img)
    len(img[0])
    print "Image row count:", len(img)
    print "Image column count:", len(img[0])
except:
    raise Exception("Image not found")

'''cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = img

# Find faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print "Crimson shape:", crimson.shape

# Draw rectangles on faces
for (x,y,w,h) in faces:
    print (x,y,w,h)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),2)

    # Draw cross on face
    #cv2.line(img,(x,y),(x+w,y+h),(0,0,255),3)
    #cv2.line(img,(x+w,y),(x,y+h),(0,0,255),3)

    # Draw smiley on top of face
    img[y:y+crimson.shape[0], x:x+crimson.shape[1]] = crimson

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        pass
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# Print result
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()