# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:57:14 2018

@author: Chandra Kant Sharma
"""

import numpy as np
import cv2

# A classifier to find faces in the image
face_cascade = cv2.CascadeClassifier(r"D:\ml_projects\ml\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Using the default camera to capture the video
cap = cv2.VideoCapture(0)

# For each frame of the video
while cap.isOpened():

	# Capture the frame image
	ret, img = cap.read()

	if ret == True:
		# Convert the captured image to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Detect faces in the grayscale image using the classifier.
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		# For each faces draw rectangle over them.
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

		# Display the image with rectangle in it.
		cv2.imshow("img", img)

		# Quit if Q is pressed
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

# Release the camera.
cap.release()

# Close all the windows.
cv2.destroyAllWindows()