# from cvzone.HandTrackingModule import HandDetector
# import cv2

import cv2
import mediapipe as mp
import math
import serial


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
                    
                    cv2.putText(img, f'{handType.classification[0].score:.2f}', (bbox[0] + 100, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
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

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.5, maxHands=2)

while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers, confidence_score = detector.fingersUp(hand1)
        if confidence_score>1:
            confidence_score = 1
        cv2.putText(img, f'{confidence_score:.2f}', (bbox1[0]+100, bbox1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        # print(fingers)

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
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

# this is the code for using teachable machines
# import cv2
# import numpy as np
# from cvzone.HandTrackingModule import HandDetector
# from keras.models import load_model

# # Load the hand detection and finger counting model
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# model = load_model('fingercount.h5', compile=False)
# labels = open('fingercountlabels.txt', 'r').readlines()

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# while True:
#     # Read the video frame
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
    
#     # Detect the hands in the frame
#     # hands = detector.findHands(img, draw=False) 
#     hands = detector.findHands(img) 
    
#     if hands:
#         # Get information of the first hand
#         hand1 = hands[0]
#         centerPoint1 = hand1['center']
#         bbox1 = hand1["bbox"]
#         f1 = detector.fingersUp(hand1)

#         # If there is a second hand, get information of that as well
#         if len(hands) == 2:
#             hand2 = hands[1]
#             centerPoint2 = hand2['center']
#             bbox2 = hand2["bbox"]
#             f2 = detector.fingersUp(hand2)

#             # Calculate distance between the two hands
#             length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

#             print(int(length), sum(f1), sum(f2))

#             # Set up box size and position for both hands
#             box_width, box_height = 175, 175
#             center_x1, center_y1 = centerPoint1
#             X1 = int(center_x1 - box_width / 2)
#             Y1 = int(center_y1 - box_height / 2)
#             W1 = box_width
#             H1 = box_height 

#             center_x2, center_y2 = centerPoint2
#             X2 = int(center_x2 - box_width / 2)
#             Y2 = int(center_y2 - box_height / 2)
#             W2 = box_width
#             H2 = box_height

#             # Store positions of both boxes
#             pos = [(X1, Y1, W1, H1), (X2, Y2, W2, H2)]
#             RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             # For each box, check if it's empty
#             for (x,y,w,h) in pos:
#                 hand = RGB[y:y+h,x:x+w]
#                 if hand.size == 0:
#                     break

#                 # Resize the hand to be processed by the model
#                 image = cv2.resize(hand, (224, 224), interpolation=cv2.INTER_AREA)
#                 image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#                 image = (image / 127.5) - 1

#                 # Get prediction from the model
#                 prediction = model.predict(image, verbose=0)
#                 index = np.argmax(prediction)
#                 class_name = labels[index]
#                 confidence_score = prediction[0][index]

#                 # Get confidence score display as a string
#                 confidence_score_display = str(np.round(confidence_score * 100, 2)) + "%"   
                
#                 # Draw bounding box and add label with class name and confidence score
#                 if length > 150:
#                     # Write "forward" text at the top right corner
#                     cv2.putText(img, "FORWARD", (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     cv2.putText(img, class_name + ": " + confidence_score_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 else:
#                     # Write "backward" text at the top right corner
#                     cv2.putText(img, "BACKWARD", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
               
                    

#     # Show output
#     cv2.imshow("Face Detection", img)

#     # Check if 'q' key is pressed to stop the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Close cam and close window
# cap.release()
# cv2.destroyAllWindows()