import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpPython
from mediapipe.tasks.python import vision as mpVision
import csv
import os
import pickle
import numpy as np

modelPath = "gesture_recognizer.task"
isCollecting = False
currentLabel = "Malevolent Shrine"
appMode = "one_hand"

if isCollecting and not os.path.exists("one_jjk_data.csv"):
    csvHeader = ["Label"]
    for i in range(21):
        csvHeader.append("x" + str(i))
        csvHeader.append("y" + str(i))
        csvHeader.append("z" + str(i))
    jjkDataFile = open("one_jjk_data.csv", "w", newline="")
    csv.writer(jjkDataFile).writerow(csvHeader)
    jjkDataFile.close()

modelOne = None
if os.path.exists("model_one_hand.pkl"):
    modelFile = open("model_one_hand.pkl", "rb")
    modelOne = pickle.load(modelFile)
    modelFile.close()
    print("Model loaded!")
else:
    print("No model found: run train.py first or set isCollecting = True to collect data")

baseOptions = mpPython.BaseOptions(model_asset_path=modelPath)

gestureOptions = mpVision.GestureRecognizerOptions(
    base_options=baseOptions,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)

gestureRecognizer = mpVision.GestureRecognizer.create_from_options(gestureOptions)

handConnections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

imgDict = {"Malevolent Shrine": "MalevolentShrine.png", "InfiniteVoid": "InfiniteVoid.png"}


def draw_landmarks(frameImage, handLandmarksList, frameHeight, frameWidth):
    for singleHandLandmarks in handLandmarksList:
        pointList = []
        for lm in singleHandLandmarks:
            pointList.append((int(lm.x * frameWidth), int(lm.y * frameHeight)))

        for startIdx, endIdx in handConnections:
            cv2.line(frameImage, pointList[startIdx], pointList[endIdx], (255, 255, 255), 2)

        for pointPos in pointList:
            cv2.circle(frameImage, pointPos, 5, (0, 255, 0), cv2.FILLED)


def get_left_hand(recognitionResult):
    for i, handCategory in enumerate(recognitionResult.handedness):
        if handCategory[0].category_name == "Left":
            return recognitionResult.hand_landmarks[i]
    return None


videoCap = cv2.VideoCapture(0)

while True:
    isSuccess, frameImg = videoCap.read()
    frameImg = cv2.flip(frameImg, 1)
    frameHeight, frameWidth, frameChannels = frameImg.shape
    rgbFrame = cv2.cvtColor(frameImg, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
    recognitionResult = gestureRecognizer.recognize(mpImage)

    if len(recognitionResult.hand_landmarks) > 0:
        draw_landmarks(frameImg, recognitionResult.hand_landmarks, frameHeight, frameWidth)

    gestureText = "No left hand detected"

    if not isCollecting and modelOne is not None:
        leftHandLandmarks = get_left_hand(recognitionResult)
        if leftHandLandmarks is not None:
            coordList = []
            for lm in leftHandLandmarks:
                coordList.extend([lm.x, lm.y, lm.z])
            predictedLabel = modelOne.predict([coordList])[0]
            if predictedLabel != "Unknown":
                gestureText = predictedLabel
            else:
                gestureText = "No gesture"

    frameImg[0:50, 0:640] = 0, 0, 0
    if isCollecting:
        cv2.putText(frameImg, "COLLECTING: " + currentLabel, (15, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frameImg, gestureText, (15, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if gestureText in imgDict:
            techniqueImage = cv2.imread(imgDict[gestureText])
            if techniqueImage is not None:
                cv2.imshow("Technique", techniqueImage)
            else:
                print("Image file not found: " + imgDict[gestureText])

    cv2.imshow("JJK Detector", frameImg)

    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord("q"):
        break

    if pressedKey == ord("s") and isCollecting:
        if appMode == "one_hand":
            leftHandLandmarks = get_left_hand(recognitionResult)
            if leftHandLandmarks is not None:
                dataRow = [currentLabel]
                for lm in leftHandLandmarks:
                    dataRow.extend([lm.x, lm.y, lm.z])
                csvAppendFile = open("one_jjk_data.csv", "a", newline="")
                csv.writer(csvAppendFile).writerow(dataRow)
                csvAppendFile.close()
                print("Saved sample for: " + currentLabel)
            else:
                print("No left hand detected")

videoCap.release()
cv2.destroyAllWindows()
