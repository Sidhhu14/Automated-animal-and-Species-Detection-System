import cv2
import numpy as np

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Create blob from input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)

    # Set input to YOLOv3 network
    net.setInput(blob)

    # Get output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Run forward pass through YOLOv3 network
    outputs = net.forward(output_layers)
    

    # Extract bounding boxes, confidences, and class IDs from network output
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on original input frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "{}: {:.2f}".format(label, confidence), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show output frame
    cv2.imshow("Object Detection", frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
