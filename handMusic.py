import os
import math
import time
import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sampleRate = 44100
targetFrequencies = [0.0, 0.0, 0.0]
currentFrequencies = [0.0, 0.0, 0.0]
phases = [0.0, 0.0, 0.0]
lastDetectionTime = 0.0
smoothingGracePeriod = 0.15

rootNotes = [
    {"name": "C", "root": 261.63},
    {"name": "D", "root": 293.66},
    {"name": "E", "root": 329.63},
    {"name": "F", "root": 349.23},
    {"name": "G", "root": 392.00},
    {"name": "A", "root": 440.00},
    {"name": "B", "root": 493.88}
]

qualities = [
    {"name": "MAJ", "third": 1.2599, "fifth": 1.4983},
    {"name": "MIN", "third": 1.1892, "fifth": 1.4983},
    {"name": "DIM", "third": 1.1892, "fifth": 1.4142},
    {"name": "AUG", "third": 1.2599, "fifth": 1.5874}
]

def audio_callback(outdata, frames, time_info, status):
    global phases, currentFrequencies
    glissandoSpeed = 0.05
    outdataBuffer = np.zeros((frames, 1), dtype=np.float32)
    
    for i in range(frames):
        if isPlaying and targetFrequencies[0] > 0:
            mixedWave = 0.0
            voiceGains = [0.75, 0.5, 0.65] 
            
            for v in range(3):
                if currentFrequencies[v] == 0:
                    currentFrequencies[v] = targetFrequencies[v]
                else:
                    currentFrequencies[v] += (targetFrequencies[v] - currentFrequencies[v]) * glissandoSpeed
                
                fWave = np.sin(2 * np.pi * phases[v])
                overtoneWave = 0.20 * np.sin(4 * np.pi * phases[v])
                
                mixedWave += (fWave + overtoneWave) * voiceGains[v]
                
                phases[v] += currentFrequencies[v] / sampleRate
                if phases[v] > 1.0:
                    phases[v] -= 1.0
            
            outdataBuffer[i] = mixedWave / 1.5
        else:
            currentFrequencies = [0.0, 0.0, 0.0]
            outdataBuffer[i] = 0.0
            
    outdata[:] = outdataBuffer

isPlaying = False
audioStream = sd.OutputStream(channels=1, samplerate=sampleRate, callback=audio_callback)
audioStream.start()

scriptDir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(scriptDir, "hand_landmarker.task")

baseOptions = mp_python.BaseOptions(model_asset_path=modelPath)
options = mp_vision.HandLandmarkerOptions(
    base_options=baseOptions,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
handTracker = mp_vision.HandLandmarker.create_from_options(options)

videoCap = cv2.VideoCapture(0)

while True:
    success, img = videoCap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    frameH, frameW, frameC = img.shape
    overlay = img.copy()
    
    outerRadius = 200
    innerRadius = 60
    
    leftCx = outerRadius + 100
    leftCy = frameH - (outerRadius + 120)
    
    rightCx = frameW - (outerRadius + 100)
    rightCy = frameH - (outerRadius + 120)
    
    activeRootIdx = -1
    activeQualityIdx = -1
    fingerInInstrument = False
    
    distL = 9999
    distR = 9999

    rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
    frameTimestampMs = int(time.time() * 1000)
    detectionResult = handTracker.detect_for_video(mpImage, frameTimestampMs)

    if detectionResult.hand_landmarks and len(detectionResult.hand_landmarks) > 0:
        for handLandmarks in detectionResult.hand_landmarks:
            indexTip = handLandmarks[8]
            pixelX = int(indexTip.x * frameW)
            pixelY = int(indexTip.y * frameH)
            
            cv2.circle(img, (pixelX, pixelY), 8, (0, 255, 0), cv2.FILLED)
            
            dlx = pixelX - leftCx
            dly = pixelY - leftCy
            currentDistL = math.sqrt(dlx**2 + dly**2)
            if currentDistL < distL:
                distL = currentDistL
            
            if innerRadius <= currentDistL <= outerRadius:
                angleL = math.degrees(math.atan2(dly, dlx)) - 270
                while angleL < 0:
                    angleL += 360
                activeRootIdx = int(angleL // (360 / 7))
                
            drx = pixelX - rightCx
            dry = pixelY - rightCy
            currentDistR = math.sqrt(drx**2 + dry**2)
            if currentDistR < distR:
                distR = currentDistR
            
            if innerRadius <= currentDistR <= outerRadius:
                angleR = math.degrees(math.atan2(dry, drx)) - 270
                while angleR < 0:
                    angleR += 360
                activeQualityIdx = int(angleR // (360 / 4))

    if activeRootIdx != -1 and activeQualityIdx != -1:
        rootF = rootNotes[activeRootIdx]["root"]
        mult3rd = qualities[activeQualityIdx]["third"]
        mult5th = qualities[activeQualityIdx]["fifth"]
        
        targetFrequencies = [rootF / 2.0, (rootF * mult3rd) * 2.0, rootF * mult5th]
        fingerInInstrument = True
        lastDetectionTime = time.time()
    else:
        if (activeRootIdx == -1 and distL < innerRadius) or (activeQualityIdx == -1 and distR < innerRadius):
            fingerInInstrument = False
            lastDetectionTime = 0.0

    if fingerInInstrument:
        isPlaying = True
    else:
        if time.time() - lastDetectionTime < smoothingGracePeriod:
            isPlaying = True
        else:
            isPlaying = False

    cv2.circle(overlay, (leftCx, leftCy), outerRadius, (20, 20, 20), -1)
    wedgeL = 360 / 7
    for i in range(7):
        startAngle = (i * wedgeL) - 90
        endAngle = ((i + 1) * wedgeL) - 90
        if i == activeRootIdx and isPlaying:
            cv2.ellipse(overlay, (leftCx, leftCy), (outerRadius, outerRadius), 0, startAngle, endAngle, (240, 240, 240), -1)
        rad = math.radians(startAngle)
        cv2.line(overlay, (int(leftCx + innerRadius * math.cos(rad)), int(leftCy + innerRadius * math.sin(rad))),
                 (int(leftCx + outerRadius * math.cos(rad)), int(leftCy + outerRadius * math.sin(rad))), (100, 100, 100), 2)
        txtAngle = math.radians(startAngle + (wedgeL / 2))
        textColor = (0, 0, 0) if (i == activeRootIdx and isPlaying) else (255, 255, 255)
        cv2.putText(overlay, rootNotes[i]["name"], (int(leftCx + (outerRadius * 0.65) * math.cos(txtAngle)) - 10, int(leftCy + (outerRadius * 0.65) * math.sin(txtAngle)) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2)
    cv2.circle(overlay, (leftCx, leftCy), outerRadius, (150, 150, 150), 3)
    cv2.circle(overlay, (leftCx, leftCy), innerRadius, (30, 30, 30), -1)
    cv2.circle(overlay, (leftCx, leftCy), innerRadius, (100, 100, 100), 2)
    cv2.putText(overlay, "OFF", (leftCx - 18, leftCy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.circle(overlay, (rightCx, rightCy), outerRadius, (20, 20, 20), -1)
    wedgeR = 360 / 4
    for i in range(4):
        startAngle = (i * wedgeR) - 90
        endAngle = ((i + 1) * wedgeR) - 90
        if i == activeQualityIdx and isPlaying:
            cv2.ellipse(overlay, (rightCx, rightCy), (outerRadius, outerRadius), 0, startAngle, endAngle, (240, 240, 240), -1)
        rad = math.radians(startAngle)
        cv2.line(overlay, (int(rightCx + innerRadius * math.cos(rad)), int(rightCy + innerRadius * math.sin(rad))),
                 (int(rightCx + outerRadius * math.cos(rad)), int(rightCy + outerRadius * math.sin(rad))), (100, 100, 100), 2)
        txtAngle = math.radians(startAngle + (wedgeR / 2))
        textColor = (0, 0, 0) if (i == activeQualityIdx and isPlaying) else (255, 255, 255)
        cv2.putText(overlay, qualities[i]["name"], (int(rightCx + (outerRadius * 0.6) * math.cos(txtAngle)) - 18, int(rightCy + (outerRadius * 0.6) * math.sin(txtAngle)) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
    cv2.circle(overlay, (rightCx, rightCy), outerRadius, (150, 150, 150), 3)
    cv2.circle(overlay, (rightCx, rightCy), innerRadius, (30, 30, 30), -1)
    cv2.circle(overlay, (rightCx, rightCy), innerRadius, (100, 100, 100), 2)
    cv2.putText(overlay, "OFF", (rightCx - 18, rightCy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.imshow("HandMusic Tracking", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

videoCap.release()
cv2.destroyAllWindows()
audioStream.stop()
audioStream.close()
