# import cv2

# cap = cv2.VideoCapture(2)
# while True:
#     suc, pic = cap.read()
#     gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
#     faceCas = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
#     faces = faceCas.detectMultiScale(gray, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(pic, (x,y),(x+w, y+h),(0,0,255),2)
#         face22 = pic[y:y + h, x:x + w]
#         cv2.imshow("face",face22)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break







import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates



# load face detection model
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)
dframe= cv2.imread('xx.png',0)
image_rows, image_cols, _ = dframe.shape[0],dframe.shape[1]
image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
results = mp_face.process(image_input)
detection=results.detections[0]
location = detection.location_data

relative_bounding_box = location.relative_bounding_box
rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
    image_rows)
rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
    image_rows)


## Lets draw a bounding box
color = (255, 0, 0)
thickness = 2
cv2.rectangle(image_input, rect_start_point, rect_end_point, color, thickness)
xleft,ytop=rect_start_point
xright,ybot=rect_end_point