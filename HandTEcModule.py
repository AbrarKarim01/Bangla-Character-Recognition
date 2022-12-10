import cv2
import mediapipe as mp
import time

# class for calling the module

class handDetector():
    def __init__(self, mode=False, maxHands=1, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        # variable of the object
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)  # creating object with library default parameter
        self.mpDraw = mp.solutions.drawing_utils  # drawing landmarks of 21 points with methods

    # class for finding hand and landmarks
    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB ( hand class only support RGB )
        self.results = self.hands.process(imgRGB)  # process the frame by receiving the object

        # Checking multiple hands
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMarks,
                                               self.mpHands.HAND_CONNECTIONS)  # drawing the landmarks with lines
        return img

    def findPosition(self, img, handNo=0, draw=True):  # checking index by finding id and LandMarks information by Using X, Y, channel Coordinates

        LandMarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, LandMark in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape  # give width, height and channels
                cx, cy, cz = int(LandMark.x * w), int(LandMark.y * h), int(LandMark.z * w)  # position of height, width pixel value from center
                print("Id: ", id, " X:", cx, " Y:", cy, " Z:", cz)

                #LandMarkList.append(['Id:', id, 'X: ', cx, 'Y: ', cy, 'Z: ', cz]) #  call when used for printing

                if id == 8:  # Detecting landmark for 4
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return LandMarkList

# Dummy
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # opening a default camera
    detector = handDetector()  # creating object
    while True:
        success, img = cap.read()  # capturing the frame rate
        img = detector.findHands(img)  # method under class
        LandMarkList = detector.findPosition(img)  # printing a specific landmark
        if len(LandMarkList) != 0:
            print(LandMarkList[0])

        # FrameRate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)  # showing frameRate

        cv2.imshow("Image", img)  # capturing the image as video
        cv2.waitKey(1)  # capturing the video


if __name__ == "__main__":
    main()