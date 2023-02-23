# from cvzone.HandTrackingModule import HandDetector
# import cv2

import cv2
import mediapipe as mp
import math
import serial
import numpy as np
from keras.models import load_model
from cvzone.ClassificationModule import Classifier

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []
        self.finger_length=[74,127,137,129,110] 

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
                    
                    # cv2.putText(img, f'{handType.classification[0].score:.2f}', (bbox[0] + 100, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):

        
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        
        if self.results.multi_hand_landmarks:
            fingers = []
            confidence_score = []
            # hand_landmarks = self.results.multi_hand_landmarks[self.hand_number]
            # confidence = hand_landmarks.visibility
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    confidence_score = abs(myLmList[self.tipIds[0]][0] - myLmList[self.tipIds[0] - 1][0]) / \
                        (self.finger_length[0]/2)
                    fingers.append(1)
                    
                else:
                    confidence_score = abs(myLmList[self.tipIds[0]][0] - myLmList[self.tipIds[0] - 1][0]) / \
                        (self.finger_length[0]/2)
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    confidence_score = abs(myLmList[self.tipIds[0]][0] - myLmList[self.tipIds[0] - 1][0]) / \
                        (self.finger_length[0]/2)
                    fingers.append(1)
                else:
                    confidence_score = abs(myLmList[self.tipIds[0]][0] - myLmList[self.tipIds[0] - 1][0]) / \
                        (self.finger_length[0]/2)
                    fingers.append(0)


            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    confidence_score = abs(myLmList[self.tipIds[id]][1] - myLmList[self.tipIds[id] - 2][1]) / \
                        (self.finger_length[id]/2)
                    fingers.append(1)
                else:
                    confidence_score = abs(myLmList[self.tipIds[id]][1] - myLmList[self.tipIds[id] - 2][1]) / \
                        (self.finger_length[id]/2)
                    fingers.append(0)
        return fingers, confidence_score
    

    

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

#initialize the video capture

cap = cv2.VideoCapture(0)

#load the models 

detector = HandDetector(detectionCon=0.5, maxHands=2)

# setting up for the white background image

offset = 20  
imgSize = 300
counter = 0 

classifier = Classifier("keras_model.h5","labels.txt")
labels = ['0', '1', '2', '3', '4', '5'] 
while True:
    # Get image frame
    success, img = cap.read()
    imgOutput = img.copy()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    
    if hands:
        # Hand 1

        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img [y-offset: y + h + offset, x - offset: x + w + offset] 
        imgWhite = np.ones((imgSize, imgSize,3), np.uint8)*255
        lmList1 = hand["lmList"]  # List of 21 Landmark points
        bbox1 = hand["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand['center']  # center of the hand cx,cy
        handType1 = hand["type"]  # Handtype Left or Right
        imgCropShape = imgCrop.shape
        

        aspectRatio = h/w

        if aspectRatio > 1:

            constant = imgSize/h
            wCal = math.ceil(constant*w)    
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap: wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            

        else:

            constant = imgSize/w
            hCal = math.ceil(constant*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap: hCal+hGap, :] 
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
           
        
        confidence_score = prediction[index]
        confidence_score_display = str(np.round(confidence_score * 100, 2)) + "%"
        cv2.putText(imgOutput, labels[index] + ": " + confidence_score_display, (x-10, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImmageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        fingers, confidence_score = detector.fingersUp(hand)

        # code for sending the serial data 

        # Set the serial port name and baud rate
        port = 'COM3'
        baudrate = 9600

        # Create a serial object
        # ser = serial.Serial(port, baudrate)

        #send an array of integers

        message = ','.join(str(x) for x in fingers) + '\n'

        print(message)
     
        # ser.write(message.encode())
        
        # if len(hands) == 2:
        #     # Hand 2
        #     hand2 = hands[1]
        #     lmList2 = hand2["lmList"]  # List of 21 Landmark points
        #     bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
        #     centerPoint2 = hand2['center']  # center of the hand cx,cy
        #     handType2 = hand2["type"]  # Hand Type "Left" or "Right"

    # Display
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
