# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# counter = 0 

# folder = "Data1/5"

# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)

#     if hands: 
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgCrop = img [y-offset: y + h + offset, x - offset: x + w + offset] 
#         imgWhite = np.ones((imgSize, imgSize,3), np.uint8)*255

#         imgCropShape = imgCrop.shape
       

#         aspectRatio = h/w

#         if aspectRatio > 1:

#             constant = imgSize/h
#             wCal = math.ceil(constant*w)    
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal)/2)
#             imgWhite[:, wGap: wCal+wGap] = imgResize
#             imgWhite = cv2.cvtColor(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)    
    
#         else:

#             constant = imgSize/w
#             hCal = math.ceil(constant*h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal)/2)
#             imgWhite[hGap: hCal+hGap, :] = imgResize
#             imgWhite = cv2.cvtColor(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)



#         cv2.imshow("ImmageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)

#     if key == ord("s"):
#         counter += 1
#         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
#         print(counter) 


# # import cv2
# # from cvzone.HandTrackingModule import HandDetector
# # import numpy as np
# # import math
# # import time

# # cap = cv2.VideoCapture(0)
# # detector = HandDetector(maxHands=1)
# # offset = 20
# # imgSize = 300
# # counter = 0 

# # folder = "Data/5"

# # while True:
# #     success, img = cap.read()
# #     hands, img = detector.findHands(img)

# #     if hands: 
# #         hand = hands[0]
# #         x, y, w, h = hand['bbox']
# #         imgCrop = img [y-offset: y + h + offset, x - offset: x + w + offset] 
# #         imgBlack = np.zeros((imgSize, imgSize, 3), np.uint8)

# #         imgCropShape = imgCrop.shape
       
# #         handLandmarks = hand["lmList"]
# #         handLandmarksArray = np.array(handLandmarks)
# #         handContour = np.array(handLandmarksArray[1:,:2], np.int32)

        

# #         aspectRatio = h/w

# #         if aspectRatio > 1:

# #             constant = imgSize/h
# #             wCal = math.ceil(constant*w)    
# #             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
# #             imgResizeShape = imgResize.shape
# #             wGap = math.ceil((imgSize - wCal)/2)
# #             imgBlack[:, wGap: wCal+wGap] = imgResize
            
# #         else:

# #             constant = imgSize/w
# #             hCal = math.ceil(constant*h)
# #             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
# #             imgResizeShape = imgResize.shape
# #             hGap = math.ceil((imgSize - hCal)/2)
# #             imgBlack[hGap: hCal+hGap, :] = imgResize

        
# #         cv2.drawContours(imgBlack, [handContour], 0, (255, 255, 255), -1)

# #         cv2.imshow("Contour", cv2.cvtColor(cv2.cvtColor(imgBlack, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))

# #         # cv2.imshow("ImmageCrop", imgCrop)
# #         cv2.imshow("ImageBlack", imgBlack)
# #     cv2.imshow("Image", img)
# #     key = cv2.waitKey(1)

# #     if key == ord("s"):
# #         counter += 1
# #         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgBlack)
# #         print(counter)

# from cvzone.HandTrackingModule import HandDetector
# import cv2

import cv2
import mediapipe as mp
import math
import numpy as np
import time

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
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                               self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=5))
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

#initialize the video capture

cap = cv2.VideoCapture(0)

#load the models 

detector = HandDetector(detectionCon=0.5, maxHands=2)

# setting up for the white background image

offset = 20  
imgSize = 300
counter = 0 
folder = "Data/5"

while True:
    # Get image frame
    success, img = cap.read()

    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) 
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
            

        else:

            constant = imgSize/w
            hCal = math.ceil(constant*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap: hCal+hGap, :] 
           
        

        cv2.imshow("ImmageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

       
    # Display
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
cap.release()
cv2.destroyAllWindows()
