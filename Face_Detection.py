import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream

# define the argument parser and parse  arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", '--prototxt', required=True,
                help="path to prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe model file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="min probability to filter weak detections")
args = vars(ap.parse_args())
print("[Information] Loading model ...")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[Information] Starting Video Stream ...")
vs = VideoStream(src=0).start()
time.sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    # grab frame dims and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract confidence
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue
        # compute x and y cords of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding boxes
        text = "{:2f}%".format((confidence * 100))
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# do a bit clean up

# do a bit clean up
cv2.destroyAllWindows()
vs.stop()
