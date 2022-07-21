from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(1)
detector = HandDetector(detectionCon=0.8, maxHands=2)
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

        fingers1 = detector.fingersUp(hand1)
        print(fingers1)
        if fingers1[1] == 1 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
            cv2.putText(img,'1',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)
        if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0 and fingers1[4] == 0:
            cv2.putText(img,'2',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)
        if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 0:
            cv2.putText(img,'3',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)
        if fingers1[0] == 0 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
            cv2.putText(img,'4',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)
        if fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
            cv2.putText(img,'0',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)
        if fingers1[0] == 1 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
            cv2.putText(img,'5',(bbox1[0] - 30, bbox1[1] - 30),cv2.FONT_HERSHEY_PLAIN,5,(25, 255, 255),5)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)
            print(type(fingers2))

            # Find D
            # istance between two Landmarks. Could be same hand or different hands
            length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()