# import cv2
# import numpy as np
# import mediapipe
#
#
# cap = cv2.VideoCapture("videos/Valorant 2022.09.05 - 23.55.24.06.DVR_Trim.mp4")
#
# while True:
#     success, img = cap.read()
#     img = cv2.GaussianBlur(img, (9, 9) , 8)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.Canny(img, 100, 100)
#
#     kernel = np.ones((5, 5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
#
#     cv2.imshow('res', img)
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
#
# img = cv2.imread("images/2021-02-14.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
# r, g, b = cv2.split(img)
# img = cv2.merge([b, g, r])
#
# cv2.imshow('res', img)
# cv2.waitKey(0)
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
#
# img = np.zeros((400, 400), dtype='uint8')
# circle = cv2.circle(img.copy(), (125, 125), 50, 255, -1)
# square = cv2.rectangle(img.copy(), (10, 10), (390 , 390), 255, -1)
#
# cv2.imshow('res', circle)
# cv2.imshow('resulr', square)
# cv2.waitKey(0)
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
#
# img = np.zeros((400, 400), dtype='uint8')
# circle = cv2.circle(img.copy(), (0, 0), 80, 255, -1)
# square = cv2.rectangle(img.copy(), (25, 25), (250 , 250), 255, -1)
#
# img = cv2.bitwise_and(circle, square)
#
# # cv2.imshow('res', circle)
# # cv2.imshow('resulr', square)
# cv2.imshow('res', img)
# cv2.waitKey(0)
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     framx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
#     framy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
#     edge = cv2.Canny(gray, 80, 150)
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('framex', framx)
#     cv2.imshow('framey', framy)
#     cv2.imshow('edge', edge)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edge = cv2.Canny(gray, 100, 200)
#     contours, h = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#
#     cv2.drawContours(frame, [contours[0]], -1,  (0, 0 , 255), 5)
#
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('edge', edge)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
# TRACKBAR
#
# import cv2
# import numpy as np
# def nothing(x):
#     pass
#
# # img = np.zeros((300, 300, 3), np.uint8)
#
# cv2.namedWindow('image')
# cv2.createTrackbar('R', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('B', 'image', 0, 255, nothing)
#
# while True:
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#
#     img[:] = [b, g, r]
#-----------------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np
# import time
#
# cap = cv2.VideoCapture(0)
#
# def nothing(x):
#     pass
# kernel = np.ones((5, 5), np.uint8)
#
#
# cv2.namedWindow('track', cv2.WINDOW_NORMAL)
# cv2.createTrackbar('H', 'track', 0, 180, nothing)
# cv2.createTrackbar('S', 'track', 0, 255, nothing)
# cv2.createTrackbar('V', 'track', 0, 255, nothing)
#
# cv2.createTrackbar('HL', 'track', 0, 180, nothing)
# cv2.createTrackbar('SL', 'track', 0, 255, nothing)
# cv2.createTrackbar('VL', 'track', 0, 255, nothing)
#
# while True:
#     ret, frame= cap.read(0)
#     cv2.imshow('frame', frame)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     h = cv2.getTrackbarPos('H', 'track')
#     s = cv2.getTrackbarPos('S', 'track')
#     v = cv2.getTrackbarPos('V', 'track')
#
#     hl = cv2.getTrackbarPos('HL', 'track')
#     sl = cv2.getTrackbarPos('SL', 'track')
#     vl = cv2.getTrackbarPos('VL', 'track')
#
#
#     lower = np.array([hl, sl, vl])
#     upper = np.array([h, s, v])
#     mask = cv2.inRange(hsv, lower, upper)
#     res = cv2.bitwise_and(frame, frame, mask = mask)
#
#
#     # erosion = cv2.erode(mask, kernel, iterations = 1)
#     # delation = cv2.dilate(mask, kernel,  iterations = 1)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening,  cv2.MORPH_CLOSE, kernel)
#     # cv2.imshow('er', erosion)
#     # cv2.imshow('del', delation)
#     cv2.imshow('open', opening)
#     cv2.imshow('closed', closing)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#-----------------------------------------------------------------------------------------------------------------------