from cvzone.HandTrackingModule import HandDetector
import cv2

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
    

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

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