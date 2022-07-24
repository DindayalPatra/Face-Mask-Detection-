# from keras.optimizers import RMSprop
# from keras.preprocessing.image import ImageDataGenerator
# import cv2
# from keras.models import Sequential
# from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
# from keras.models import Model, load_model
# from keras.callbacks import TensorBoard, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from sklearn.utils import shuffle
# import imutils
# import numpy as np

# model = Sequential([
#     Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2,2),
    
#     Conv2D(100, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
    
#     Flatten(),
#     Dropout(0.5),
#     Dense(50, activation='relu'),
#     Dense(2, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# TRAINING_DIR = "C:/Users/BISWAJIT/Downloads/face-mask-dataset/Dataset/train/train/"
# train_datagen = ImageDataGenerator(rescale=1.0/255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

# train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
#                                                     batch_size=10, 
#                                                     target_size=(150, 150))
# VALIDATION_DIR = "C:/Users/BISWAJIT/Downloads/face-mask-dataset/Dataset/test/test/"
# validation_datagen = ImageDataGenerator(rescale=1.0/255)

# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
#                                                          batch_size=10, 
#                                                          target_size=(150, 150))

# checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

# history = model.fit_generator(train_generator,
#                               epochs=10,
#                               validation_data=validation_generator,
#                               callbacks=[checkpoint])

# model.save('A:/work space/python/maskmodel.h5')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import imutils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR=1e-4
EPOCHS =10
BS =32

Direc=r'F:\mask_detection\Image_data'
Cata=["with_mask","without_mask"]

data=[]
lab=[]
for catagory in Cata:
    path=os.path.join(Direc,catagory)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)
        data.append(image)
        lab.append(catagory)

lb=LabelBinarizer()
labels=lb.fit_transform(lab)
labels=to_categorical(labels)
data=np.array(data,dtype="float32")
labels=np.array(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)

aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

basemodel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headmodel=basemodel.output
headmodel=AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel=Flatten(name="flatten")(headmodel)
headmodel=Dense(128,activation="relu")(headmodel)
headmodel=Dense(2,activation="softmax")(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)

for layer in basemodel.layers:
    layer.trainable=False

opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

history=model.fit(aug.flow(trainX,trainY,batch_size=BS),steps_per_epoch=len(trainX)//BS,validation_data=(testX,testY),validation_steps=len(testX)//BS,epochs=EPOCHS)
predixs=model.predict(testX,batch_size=BS)
predixs=np.argmax(predixs,axis=1)
model.save(r'F:\mask_detection\Model_Output/maskmodel.h5')
print("done")
