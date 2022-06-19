import cv2
import time
from faceDetector import FaceDetector
from faceRecogniser import Classifier

cap = cv2.VideoCapture(2)
detector = FaceDetector()
classifier = Classifier('model/keras_model.h5', 'model/labels.txt')
color = (0,255,0)
while True:
    _, img = cap.read()
    bboxs = detector.findFaces(img)
    # predection = classifier.getPrediction(img)
    if bboxs:
        print( True)
    else:
        print(False)
    predection = classifier.getPrediction(img)[0]
    accuracy = "{:.2f}".format(predection[0]*100)
    cv2.putText(img, f'{int(predection[0] * 100)}%', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    print(predection[0]*100)
    print(accuracy)
    cv2.imshow("Image", img)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
