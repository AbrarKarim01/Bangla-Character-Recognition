import cv2
import os

from cvzone.HandTrackingModule import HandDetector

# Parameters
width, height = 1280, 720
gestureThreshold = 300


# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

