import cv2
import mediapipe as mp
import time
from faceRecogniser import Classifier

cap = cv2.VideoCapture(2)
mp_face_detection = mp.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection( min_detection_confidence = 0.6)
classifier = Classifier('model/keras_model.h5', 'model/labels.txt')
color = (0,255,0)
i = 0
while True:
    _, img = cap.read()
    results = face_detector.process(img)
    # if results.detections:
    for face in results.detections:
        confidence = face.score
        bounding_box = face.location_data.relative_bounding_box
        x = int(bounding_box.xmin * img.shape[1])
        w = int(bounding_box.width * img.shape[1])
        y = int(bounding_box.ymin * img.shape[0])
        h = int(bounding_box.height * img.shape[0])
        # cv2.rectangle(img, (x, y-20), (x + w, y + h), (255, 255, 255), thickness = 2)
        # cv2.imshow("Image",img)
        croppedFace = img[y-20:y + h, x:x + w]
        # cv2.imshow("Face",croppedFace)
        # cv2.imwrite(f"Frame{str(i)}.jpg", croppedFace)
        # i += 1
        # time.sleep(0.09)
        predection = classifier.getPrediction(croppedFace)[0]
        # accuracy = "{:.2f}".format(predection[0]*100)
        # cv2.putText(img, f'{int(predection[0] * 100)}%', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        print(predection[0]*100)
        cv2.imshow("Image", croppedFace)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break