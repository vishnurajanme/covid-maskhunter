#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:43:34 2020

@author: vishnurajan
"""


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from playsound import playsound
import serial
from ttkthemes import ThemedTk
confi = 0 #For Tkinter
size = 0 #For Tkinter

def detect_and_predict_mask(frame, faceNet, maskNet):
	confi = float(name_entry.get())/10
	size = int(date_entry.get())

	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > confi:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if len(face) < size:
			    print("Alert!!! Face Captured is too small to analyze. Skipping face.")
			    break
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)





# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "neural.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"neural.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("neural.model")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
#vs = VideoStream(src=0).start()
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.)
time.sleep(2.0)


import PIL
from PIL import Image,ImageTk
#import pytesseract
import cv2
import tkinter as tk
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!




root=ThemedTk(theme="breeze")
root.title('Covid Maskhunt - Automated Mask Detection & Analysis System by Dr. Vishnu Rajan. drvishnurajan.wordpress.com')
#root.bind('<Escape>', lambda e: root.quit())
width, height = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry('%dx%d+0+0' % (width,height))
root.attributes('-fullscreen', True)
width2 = int(width*80/100)
height2 = int(height*80/100)



# creating a label for  
# name using widget Label 
name_label = ttk.Label(root, text = 'Confidence', 
                      font=('calibre', 
                            11), wraplength=100) 
   
# creating a entry for input 
# name using widget Entry 
#name_entry = tk.Entry(root ,font=('calibre',10,'normal')) 
name_entry = ttk.Scale(root, from_=1, to=9, orient=tk.HORIZONTAL, value = 2)

      
# creating a label for  
# dates using widget Label 
date_label = ttk.Label(root, text = 'Min. Face Size', 
                      font=('calibre', 
                            11), wraplength=100) 
   
# creating a entry for input 
# dates using widget Entry 
#date_entry = tk.Entry(root ,font=('calibre',10,'normal')) 
date_entry = ttk.Scale(root, from_=3, to=300, orient=tk.HORIZONTAL, value = 10)

inst_label = ttk.Label(root, text = 'Instructions', 
                      font=('calibre', 
                            15)) 

inst_title = ttk.Label(root, text = 'COVID MASKHUNTER V2.0', 
                      font=('calibre', 
                            25)) 


simple_label = ttk.Label(root, text = 'Lower Confidence Levels and Min. Face size will increase the sensitivity but in turn reduce the accuracy or throw in some errors. So use the confidence and min. face size slider wisely. Use checkboxes to adjust the visual/audio alerts. To quit the program, use Alt-F4 in windows. Library requirements: Tensorflow, Tkinter, PySerial, Numpy, Python version == 3.6', 
                      font=('calibre', 
                            10), wraplength=int(width*7/30)) 

chk_box = ttk.Checkbutton(root, text="Face Bounding Box")
chk_box1 = ttk.Checkbutton(root, text="Face Analysis")
chk_box2 = ttk.Checkbutton(root, text="Warning Message")
chk_box3 = ttk.Checkbutton(root, text="Warning Audio")


  
# Button that will call the submit function  
sub_btn=ttk.Button(root,text = 'Start Analysis')

close_btn=ttk.Button(text = "Quit", command = root.destroy)



img = Image.open("splash.png")
img = img.resize((int(width/2), height-height2), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = ttk.Label(root, image = img)


splashr = "sr1.png"
img1 = Image.open(splashr)
img1 = img1.resize((width-width2, height2), Image.ANTIALIAS)
img1 = ImageTk.PhotoImage(img1)
panelr = ttk.Label(root, image = img1)


lmain = tk.Label(root)

cou = 0
color1 = (255,0,0)
color2 = (0, 204, 204)
a,b,c = 0,0,0
status = "Initializing"
m, nm, t = 0,0,0



def show_frame():
    # grab the frame from the threaded video stream and resize it
   	start = time.time()
   	global t,a,m,b, c,nm,status,color1,color2, width2, height2, splashr, panelr
#ser = serial.Serial('/dev/tty.usbserial', 9600)

   	#frame = vs.read()
   	ret, frame = vs.read()
   	#frame = cv2.QueryFrame(frame)
   	#frame = imutils.resize(frame, height = height2)
   	frame = cv2.resize(frame, (width2,height2))
   	height1, width1, dim = (frame.shape)
   	overlay = frame.copy()
   
   	# detect faces in the frame and determine if they are wearing a
   	# face mask or not
   	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
   	c += 1
   	# loop over the detected face locations and their corresponding
   	# locations
   	for (box, pred) in zip(locs, preds):
   		# unpack the bounding box and predictions
   		
   		t += 1
   		(startX, startY, endX, endY) = box
   		(mask, withoutMask) = pred
   
   		# determine the class label and color we'll use to draw
   		# the bounding box and text
   		label = "Mask found" if mask > withoutMask else "No Mask found"
   		color = (0, 255, 0) if label == "Mask found" else (0, 0, 255)
   		
           
   		if (label == "Mask found"):
   		    a += 1
   		    m += 1
   		else:
   		    b += 1
   		    nm += 1
   
   		# include the probability in the label
   		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
   
   		# display the label and bounding box rectangle on the output
   		if (chk_box1.instate(['selected']) == True):
   		       cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
   		
   		if (chk_box.instate(['selected']) == True):
   		       cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

   	if m>0:    
           if (a/m)>(15):
       	    status = "Status: Everyone found wearing Masks"
       	    color2 = (51, 204, 51)
       	    color1 = (0, 255, 0)
       	    a,b,c = 0,0,0
       	    splashr = "sr3.png"
       	    img1 = Image.open(splashr)
       	    img1 = img1.resize((width-width2, height2), Image.ANTIALIAS)
       	    img1 = ImageTk.PhotoImage(img1)
       	    panelr.configure(image=img1)
       	    panelr.Image = img1
       	    #ser.write(b'0')
   	if nm>0:
           if (b/nm)>(10):
       	    status = "Status: Persons without Mask Found"
       	    color1 = (0, 0, 255)
       	    color2 = (0, 51, 255)
       	    a,b,c = 0,0,0
       	    alpha = 0.7
       	    splashr = "sr2.png"
       	    img1 = Image.open(splashr)
       	    img1 = img1.resize((width-width2, height2), Image.ANTIALIAS)
       	    img1 = ImageTk.PhotoImage(img1)
       	    panelr.configure(image=img1)
       	    panelr.Image = img1
            if (chk_box2.instate(['selected']) == True):
           	    cv2.rectangle(frame, (0, 0), (width1, height1), (0, 0, 255), -1)
           	    #cv2.rectangle(frame, (0, 0), (600, 500), (0, 0, 255), -1)
           	    cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)
           	    #ser.write(b'1')
           	    cv2.putText(frame, "W A R N I N G", (int(width1*1/5), int(height1*3/5)), cv2.FONT_HERSHEY_SIMPLEX, 3, color1, 1)
       	    if (chk_box3.instate(['selected']) == True):
                   playsound("mal.m4a", False)
   	if c>30:
   	    status = "Status: Waiting to Detect Faces"
   	    color1 = (255, 0, 0)
   	    color2 = (255, 153, 0)
   	    a,b,c = 0,0,0
   	    splashr = "sr1.png"
   	    img1 = Image.open(splashr)
   	    img1 = img1.resize((width-width2, height2), Image.ANTIALIAS)
   	    img1 = ImageTk.PhotoImage(img1)
   	    panelr.configure(image=img1)
   	    panelr.Image = img1
   	cv2.rectangle(frame, (0, int(height1*97/100)), (width1, height1), color2, -1)
   	#cv2.putText(frame, "  Debug:   M: " + str(a) + "  NM: " + str(b) + "  NF: " + str(c), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color1, 1)
   	end = time.time()
   	seconds = end - start
   	fps  = 1 / seconds;
   	cv2.putText(frame, " Total Persons: " + str(t) +"      Faces without Mask: "+ str(nm)+ "      Faces with Mask: " + str(m) + "      " + status + "  Debug:   M: " + str(a) + "  NM: " + str(b) + "  NF: " + str(c) + " fps: " + str(round(fps, 2)), (0, int(height1*99/100)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
   	m, nm, t = 0,0,0
   	key = cv2.waitKey(1) & 0xFF
    
   	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
   	img = PIL.Image.fromarray(cv2image)
   	imgtk = ImageTk.PhotoImage(image=img)
   	lmain.imgtk = imgtk
   	lmain.configure(image=imgtk)
   	flag = 1
   	lmain.after(10, show_frame)
    
# placing the label and entry in 
# the required position using grid 
# method 
lmain.grid(row = 0, column = 0, columnspan = 5,rowspan = 1, padx = 0, pady = 0, sticky = tk.N) 
panelr.grid(row = 0, column = 5, columnspan = 1,rowspan = 1, padx = 0, pady = 0, sticky = tk.N) 
name_label.grid(row=1,column=0, padx = 0, pady = 0) 
name_entry.grid(row=2,column=0, padx = 0, pady = 0) 
date_label.grid(row=3,column=0, padx = 0, pady = 0) 
date_entry.grid(row=4,column=0, padx = 0, pady = 0) 
chk_box.grid(row=1,column=1, padx = 0, pady = 0, sticky = tk.W) 
chk_box1.grid(row=2,column=1, padx = 0, pady = 0, sticky = tk.W) 
chk_box2.grid(row=3,column=1, padx = 0, pady = 0, sticky = tk.W) 
chk_box3.grid(row=4,column=1, padx = 0, pady = 0, sticky = tk.W) 
inst_label.grid(row=1,column=2, padx = 0, pady = 0) 
simple_label.grid(row = 2, column = 2, columnspan = 1,rowspan = 3, padx = 30, pady = 0, sticky = tk.N+tk.S+tk.E+tk.W) 
panel.grid(row = 1, column = 3, columnspan = 3, rowspan = 4, padx = 0, pady = 0, sticky = tk.N+tk.S+tk.E+tk.W)

show_frame()

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)


root.mainloop()