# import by installing packages:
import cv2
import mediapipe as mp
import time

# Running a webCam


# Objects
cap = cv2.VideoCapture(0)  # opening a default camera
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # creating object with library default parameter
mpDraw = mp.solutions.drawing_utils  # drawing landmarks of 21 points with methods

# FrameRate
pTime = 0
Ctime = 0

while True:

    success, img = cap.read()  # capturing the frame rate
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB ( hand class only support RGB )
    results = hands.process(imgRGB)  # process the frame by receiving the object

    # Checking multiple hands
    if results.multi_hand_landmarks:

        # extracting multiple hands
        for handLandmarks in results.multi_hand_landmarks:
            # checking index by finding id and LandMarks information by Using X, Y, Z Coordinates
            for id, LandMark in enumerate(handLandmarks.landmark):
                h, w, c = img.shape  # give width, height and channels
                cx, cy = int(LandMark.x * w), int(LandMark.y * h)  # position of height, weight pixel value from center
                print("Id:", id, "X:", cx, "Y:", cy)

                if id == 4:  # Detecting landmark for 4
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)  # drawing the landmarks with lines

    # FrameRate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 255), 3)  # showing frameRate
    cv2.imshow("Image", img)  # capturing the image as video
    cv2.waitKey(1)  # capturing the frame rate
