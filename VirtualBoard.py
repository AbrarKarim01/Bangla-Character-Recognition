import cv2
import numpy as np
import time
import os
import HandTEcGestureModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (119, 207, 157)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hands
    img = detector.findHands(img)
    LandMarkList = detector.findPosition(img, draw=False)

    if len(LandMarkList) != 0:

        x1, y1 = LandMarkList[8][1:]
        x2, y2 = LandMarkList[12][1:]


        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

    # 4. If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 300 < x1 < 500:
                    header = overlayList[0]
                    drawColor = (119, 207, 157)
                elif 600 < x1 < 800:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 850 < x1 < 1000:
                    header = overlayList[2]
                    drawColor = (144, 128, 128)
                elif 1075 < x1 < 1250:
                    header = overlayList[3]
                    drawColor = (140, 128, 0)



    # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 25, drawColor, cv2.FILLED)
            print("Drawing Mode")

    # Setting the header image
    img[0:125,0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
