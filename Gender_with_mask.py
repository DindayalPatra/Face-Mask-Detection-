import cv2
import numpy as np
from keras.models import load_model
gendermodel=load_model(r"F:\mask_detection\Model_Output/gendermodel.h5")
maskmodel=load_model(r"F:\mask_detection\Model_Output/maskmodel.h5")

genderresults={1:'Male',0:'Female'}
maskresults={1:'Without_Mask',0:'With_Mask'}
GR_dict={0:(0,255,0),1:(0,0,255)}

rect_size = 4
cap = cv2.VideoCapture(0) 


haarcascade = cv2.CascadeClassifier(r'F:\mask_detection\Harcase_cade/haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 

    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(224,224))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        maskresult=maskmodel.predict(reshaped)
        genderresult=gendermodel.predict(reshaped)

        
        masklabel=np.argmax(maskresult,axis=1)[0]
        genderlabel=np.argmax(genderresult,axis=1)[0]
        tag=(maskresults[masklabel]+" , "+ genderresults[genderlabel])
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[genderlabel],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[genderlabel],-1)
        cv2.putText(im,tag, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('LIVE',   im)
    
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()