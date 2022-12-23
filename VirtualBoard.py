import cv2
import numpy as np
import os
import HandTEcGestureModule as htm

###########################
brushThickness = 10
eraserThickness = 50
##########################
counter= 0
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

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

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
            # xp, yp = 0, 0

            print("Selection Mode  ", "X: ", x2, "Y: ", y2)
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
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


    # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 25, drawColor, cv2.FILLED)
            print("Drawing Mode  ", "X: ", x1, "Y: ", y1)

            counter = counter + 1
            print(counter)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125,0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
