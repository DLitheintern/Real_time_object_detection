import cv2
import numpy as np
# Load YOLOv3 weights and configuration file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set input image size
input_size = 416

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (input_size, input_size), (0, 0, 0), swapRB=True, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Get output layers
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer
    for output in layerOutputs:
        # Loop over each detection
        for detection in output:
            # Get class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                # Get box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                # Add detected object to lists
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes and labels on frame for detected objects
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, classes[class_ids[i]], (x, y - 5), font, 1, color, 2)

    # Show frame
    cv2.imshow("Object Detection", frame)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == 27:
        # If ESC is pressed, break loop
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
