import cv2
import os
import mediapipe as mp
import numpy as np
import math
from PIL import Image


class HandDetector:

    def __init__(self, mode=False, maximum_hands=1, complexity=1, detection_confidence=0.85, tracking_confidence=0.85):
        self.results = None
        self.mode = mode
        self.maximum_hands = maximum_hands
        self.detection_confidence = detection_confidence
        self.complexity = complexity
        self.tracking_confidence = tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maximum_hands, self.complexity, self.detection_confidence,
                                        self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, canvas, draw=True):
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(canvas, handLms, self.mpHands.HAND_CONNECTIONS)
        return canvas

    def findPosition(self, canvas, hand_num=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = canvas.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append([id, cx, cy])

        return self.lmList

    def fingersUp(self):
        fingers_up_down = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] - 2:
            fingers_up_down.append(1)
        else:
            fingers_up_down.append(0)

        # Fingers
        for number in range(1, 5):

            if self.lmList[self.tipIds[number]][2] < self.lmList[self.tipIds[number] - 2][2]:
                fingers_up_down.append(1)
            else:
                fingers_up_down.append(0)

        return fingers_up_down


filePath = "Header"
myList = os.listdir(filePath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{filePath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector()
defColor = (20, 150, 0)
xp, yp = 0, 0

drawCanvas = np.zeros((720, 1280, 3), np.uint8)

defThickness = 20
brushThickness = defThickness

while True:

    # Initialize Image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.GaussianBlur(img, (17, 17), cv2.BORDER_DEFAULT)

    # Hand Coordinates
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # xp, yp = 0, 0
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[4][1:]
        x4, y4 = (x1 + x3) // 2, (y1 + y3) // 2

        # Checking fingers up or not
        fingers = detector.fingersUp()
        # print(fingers)

        # Selection Tool
        if fingers[1] and fingers[2]:
            # cv2.rectangle(img, (x1, y1-25), (x2, y2+25), defColor, cv2.FILLED)
            xp, yp = 0, 0

            if y1 < 100:
                if 0 < x1 < 183:
                    header = overlayList[0]
                    defColor = (20, 150, 0)
                    brushThickness = defThickness

                elif 183 < x1 < 366:
                    header = overlayList[1]
                    defColor = (0, 50, 255)
                    brushThickness = defThickness

                elif 366 < x1 < 549:
                    header = overlayList[2]
                    defColor = (210, 50, 0)
                    brushThickness = defThickness

                elif 549 < x1 < 732:
                    header = overlayList[3]
                    defColor = (7, 210, 252)
                    brushThickness = defThickness

                elif 732 < x1 < 915:
                    header = overlayList[4]
                    defColor = (102, 0, 255)
                    brushThickness = defThickness

                elif 915 < x1 < 1098:
                    header = overlayList[5]
                    defColor = (255, 255, 255)
                    brushThickness = defThickness

                elif x1 > 1098:
                    header = overlayList[6]
                    defColor = (0, 0, 0)
                    brushThickness = 80

        # Clear Screen
        if fingers[1] and fingers[2] and fingers[3]:
            if x1 > 1098 and y1 < 100:
                cv2.rectangle(drawCanvas, (0, 0), (1280, 720), (0, 0, 0), -1)

        # Drawing Tool
        if fingers[1] and not fingers[2] and not fingers[0]:
            cv2.circle(img, (x1, y1), int(brushThickness // 2), defColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), defColor, brushThickness)
            cv2.line(drawCanvas, (xp, yp), (x1, y1), defColor, brushThickness)
            xp, yp = x1, y1

        # Brush Thickness Tool
        if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            cv2.circle(img, (x1, y1), 15, (0, 0, 40), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 40), cv2.FILLED)
            cv2.line(img, (x1, y1), (x3, y3), (0, 0, 75), 10)

            length = math.hypot(x3 - x1, y3 - y1)
            brushThickness = int(length / 5)

            cv2.circle(img, (x4, y4), int(brushThickness // 1.25), defColor, cv2.FILLED)

    img[0:100, 0:1280] = header
    imgGray = cv2.cvtColor(drawCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, drawCanvas)

    capture = cv2.cvtColor(drawCanvas, cv2.COLOR_BGR2RGB)

    cv2.imshow('Air Canvas', img)

    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 99:
        im = Image.fromarray(capture, "RGB")
        im = im.save("Capture.jpg")

cv2.destroyAllWindows()
