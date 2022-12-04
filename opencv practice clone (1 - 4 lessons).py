import cv2
import numpy as np
import mediapipe

# show photo
"""
img = cv2.imread('images/gl.png')
cv2.imshow('result', img)

cv2.waitKey(10000)
"""
"""
# show videocamera
cap = cv2.VideoCapture('videos\Valorant 2022.09.05 - 23.55.24.06.DVR_Trim.mp4')




cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 500)


while True:
     success, img = cap.read()
     cv2.imshow('result', img)

     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
"""
'''
img = cv2.imread('images/gl.png')
img = cv2.resize(img, (300, 500)) # size
img = cv2.GaussianBlur(img, (9, 9), 0) # Blur

cv2.imshow('result', img)

# cv2.imshow('result', img[0:100, 0:150]) # cut

cv2.waitKey(10000)
'''
"""
cap = cv2.VideoCapture('videos/League of Legends 2022.10.15 - 11.15.23.02.DVR_Trim.mp4')
cap.set(3, 100)
cap.set(4, 100)

while True:
     success, img = cap.read()
     cv2.imshow('result', img)

     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
"""

# find angles
"""
img = cv2.imread('images/gl.png')
img = cv2.Canny(img, 90, 90)

cv2.imshow('result', img)
cv2.waitKey(10000)
"""
"""
# Изменение обводки
img = cv2.imread('images/gl.png')
kernel = np.ones((5, 5), np.uint8) # uint8 => Натуральные числа
img = cv2.dilate(img, kernel, iterations=1)


cv2.imshow('result', img)
cv2.waitKey(10000)
"""

"""
# creating img and rectangle
photo = np.zeros((300, 300, 3), dtype='uint8') # zeros => матрица
#photo[:] = 242, 101, 88 # coloring all elem
#photo[20:120, 20:120] = 242, 101, 88
cv2.rectangle(photo, (10, 10), (100, 100), (242, 101, 88), thickness=cv2.FILLED) # квадрат 1 - отступ, 2 - до какого значения, 3 - обводка

cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[0] // 2), (100, 100, 100), thickness=10)
cv2.imshow('', photo)
cv2.waitKey(0)
"""


"""
# circle
photo = np.zeros((300, 300, 3), dtype='uint8')
cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 50, (242, 101, 88), thickness= 10 )
cv2.imshow('', photo)
cv2.waitKey(0)
"""
"""
# text
photo = np.zeros((300, 300, 3), dtype='uint8')
cv2.putText(photo, 'prog', (100, 150), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
cv2.imshow('', photo)
cv2.waitKey(0)
"""


"""
cap = cv2.VideoCapture('videos/Valorant 2022.09.05 - 23.55.24.06.DVR_Trim.mp4')
while True:
     success, img = cap.read()

     img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
     img = cv2.GaussianBlur(img, (9, 9), 0)
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     img = cv2.Canny(img, 70, 40)
     kernel = np.ones((5, 5), np.uint8)
     img = cv2.dilate(img, kernel, iterations=1)

     img = cv2.erode(img, kernel, iterations=1)
     cv2.imshow('res', img)

     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
"""



"""
from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
     success, img = cap.read()
     img, bboxs = detector.findFaces(img)

     if bboxs:
        # bboxInfo - "id","bbox","score","center"
          center = bboxs[0]["center"]
          cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

     cv2.imshow("Image", img)
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break
cv2.destroyAllWindows()
cap.release()
"""
"""
cap = cv2.VideoCapture("videos/Valorant 2022.09.05 - 23.55.24.06.DVR_Trim.mp4")

img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
img = cv2.GaussianBlur(img, (9, 9), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.Canny(img, 200, 200)

kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations = 1)

img = cv2.erode(img, kernel, iterations = 1)

cv2.imshow('res', img)
cv2.waitKey(0)
"""