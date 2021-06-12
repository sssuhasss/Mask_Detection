from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import gradio as gr
from pathlib import Path


prototxtPath = str(Path('deploy.prototxt'))
weightsPath = str(Path('res10_300x300_ssd_iter_140000.caffemodel'))
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model('mask_detection.h5')


def detect_and_predict_mask(frame):
    mask_count=0
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model('mask_detection.h5')
    
	# grab the dimensions of the frame and then construct a blob
	# from it
    (h, w) = (400,400)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    class_ids=[]
    W=None
    H=None
    for i in range(0, detections.shape[2]):
        
		# extract the confidence (i.e., probability) associated with
		# the detection
        confidence = detections[0, 0, i, 2]
        class_id=int(detections[0, 0, i, 1])
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        if confidence > 0.45 :
            class_id=int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
			# add the face and bounding boxes to their respective
			# lists
            faces.append(face)
            class_ids.append(class_id)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (withoutMask, mask) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
        if mask < withoutMask:
            label = "No Mask"
            
        else:
            label="Mask"
            mask_count +=1
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    #Add top-border to frame to display stats
        border_size=50
        border_text_color=[255,255,255]
        frame = cv2.copyMakeBorder(frame, border_size,0,0,0, cv2.BORDER_CONSTANT)
    #calculate count values
        
        total_face= len(class_ids)
        
        nomask_count=total_face-mask_count
    #display count
        text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
        cv2.putText(frame,text, (0, int(border_size-5)), cv2.FONT_HERSHEY_SIMPLEX,0.4,border_text_color, 1)
    #display status
        text = "Status:"
        cv2.putText(frame,text, (W-75, int(border_size-30)), cv2.FONT_HERSHEY_SIMPLEX,0.3,border_text_color, 1)
        ratio=nomask_count/(mask_count+nomask_count+0.000001)

    
        if ratio>=0.1 and nomask_count>=3:
            text = "Danger !"
            cv2.putText(frame,text, (W-75, int(border_size-10)), cv2.FONT_HERSHEY_SIMPLEX,0.4,[26,13,247], 1)
           # if fps._numFrames>=next_frame_towait: #to send danger sms again,only after skipping few seconds
               # msg="**Face Mask System Alert** %0A%0A"
               # msg+="Camera ID: C001 %0A%0A"            
              #  msg+="Status: Danger! %0A%0A"
              #  msg+="No_Mask Count: "+str(nomask_count)+" %0A"
               # msg+="Mask Count: "+str(mask_count)+" %0A"
               # datetime_ist = datetime.now(IST) 
               # msg+="Date-Time of alert: %0A"+datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')
                #sendSMS(msg,[7041677471])
                #print('Sms sent')
                #next_frame_towait=fps._numFrames+(5*25)
        
        elif ratio!=0 and np.isnan(ratio)!=True:
            text = "Warning !"
            cv2.putText(frame,text, (W-75, int(border_size-10)), cv2.FONT_HERSHEY_SIMPLEX,0.4,[0,255,255], 1)

        else:
            text = "Safe "
            cv2.putText(frame,text, (W-75, int(border_size-10)), cv2.FONT_HERSHEY_SIMPLEX,0.4,[0,255,0], 1)
    

            
   
    return frame



webcam = gr.inputs.Image(shape=(400,400), source="webcam")
#img= gr.outputs.Image()
gr.Interface(fn=detect_and_predict_mask, inputs=webcam ,outputs="image").launch()