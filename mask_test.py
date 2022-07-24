from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
# load the model
model = load_model(r"F:\mask_detection\Model_Output/maskmodel.h5")
CATEGORIES = ["with mask","without_mask"]
cap = cv2.VideoCapture(0)
# Define mediapipe Face detector
face_detection = mp.solutions.face_detection.FaceDetection()
# Detection function for Face Mask Detection
def get_detection(frame):
    
    height, width, channel = frame.shape

  # Convert frame BGR to RGB colorspace
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Detect results from the frame
    result = face_detection.process(imgRGB)

  # Extract data from result
    try:
      for count, detection in enumerate(result.detections):
          # print(detection)
     # Extract bounding box information
          box = detection.location_data.relative_bounding_box
          x, y, w, h = int(box.xmin*width), int(box.ymin * height),int(box.width*width), int(box.height*height)
          
  # If detection is not available then pass
    except Exception as e:
         print(e)

    return x, y, w, h

while True:
  _, frame = cap.read()
  img = frame.copy()
  try:
      
      x, y, w, h = get_detection(frame)
      imgcrop = img[y:y+h, x:x+w]
      rerect_sized=cv2.resize(imgcrop,(224,224))
      normalized=rerect_sized/255.0
      reshaped=np.reshape(normalized,(1,224,224,3))
      reshaped = np.vstack([reshaped])
      result=model.predict(reshaped)
      # crop_img = cv2.resize(imgcrop, (224, 224))
      # crop_img = np.expand_dims(crop_img, axis=0)
      # prediction = model.predict(crop_img)
      # print(result)
      index = np.argmax(result)
      res = CATEGORIES[index]
     
      if index == 0:
            color = (0, 0, 255)
      else:
            color = (0, 255, 0)
      cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
      cv2.putText(frame, res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2, cv2.LINE_AA)
  except Exception as ee:

        print(ee)
    
  cv2.imshow("Live", frame)
  # cv2.imshow("Live",crop_img1 )
  if cv2.waitKey(1) == ord('q'):
        break


