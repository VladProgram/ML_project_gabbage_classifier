import cv2
import numpy as np
import argparse
import os
import time

# Load the YOLOv5 model
# TODO: check error
model = cv2.dnn.readNet('../trained_models/YOLO/yolov5s.onnx')

# Define the labels for the classes we want to detect
class_names = ['cabbage', 'weed']

# Define the colors for the bounding boxes
colors = [(0, 255, 0), (0, 0, 255)]

# Set up the input image size
input_size = (640, 640)

# Define the confidence threshold for detection
confidence_threshold = 0.5

# Define the non-maximum suppression threshold
nms_threshold = 0.4

# Define the input image file
input_file = '../data/frames/1.jpg'

# Load the input image
image = cv2.imread(input_file)

# Resize the image to the input size
image = cv2.resize(image, input_size)

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=input_size, mean=(0, 0, 0), swapRB=True, crop=False)

# Set the input to the model
model.setInput(blob)

# Perform the forward pass and get the output
output = model.forward()

# Loop over the detected objects
for detection in output[0, 0, :, :]:
    # Extract the class ID and confidence
    class_id = int(detection[1])
    confidence = detection[2]

    # Check if the detection meets the confidence threshold
    if confidence > confidence_threshold:
        # Get the coordinates of the bounding box
        x1 = int(detection[3] * image.shape[1])
        y1 = int(detection[4] * image.shape[0])
        x2 = int(detection[5] * image.shape[1])
        y2 = int(detection[6] * image.shape[0])

        # Apply non-maximum suppression to remove overlapping boxes
        scores = detection[2:]
        boxes = [detection[3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]) for detection in output[0, 0, :, :] if detection[2] > confidence_threshold]
        boxes, scores, class_ids = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)

        # Draw the bounding box and label on the image
        color = colors[class_id]
        label = class_names[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
