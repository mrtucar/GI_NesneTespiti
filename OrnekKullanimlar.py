# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:41:21 2025

@author: murat.ucar
"""

#%%
#Gerekli paketlerin kurulumu için Anaconda Prompt penceresinden aşağıdaki komutu çalıştırınız.
# Install the ultralytics package using conda
# conda install -c conda-forge ultralytics

#%%
#Bir görüntü üzerinde test ediyoruz..
from ultralytics import YOLO
model = YOLO(r"yolo11n.pt")
model.predict(source="Data/los_angeles.mp4",show=True)

#%%
#Bir görüntü üzerinde test ediyoruz. 

from ultralytics import YOLO
model = YOLO("yolo11n.pt")

results = model("Data/cat_dogs.jpeg")

for r in results:
    i = 1
    for box in r.boxes:
        box = box.numpy()
        print(str(i) + ". Nesne =>" + str(box.cls[0]))
        print(str(i) + ". Nesne Adı =>" + str(r.names[ int( box.cls[0]) ] ))
        print(box.conf )
        print(box.xywh)
        print(box.xyxy)
        print(box.xywhn)
        print(box.xyxyn)        
        print("-"*30)
        i+=1
    r.show()
    r.save(filename="sonuc.jpg")
#%%

from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
model = YOLO("yolo11n.pt")


img = cv2.imread("Data/cat_dogs.jpeg")
results = model(img)

for r in results:
    i = 1
    annotator = Annotator(img)
    
    for box in r.boxes:
        if (box.cls==16):
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b,"aranan nesne",color=(255, 0,0) )
        i+=1
    #r.save(filename="sonuc.jpg")
    cv2.imshow("Sonuc", img)
    cv2.imwrite("sonuc.jpg", img)
  
#%%
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
model = YOLO("yolo11n.pt")

img = cv2.imread("Data/cat_dogs.jpeg")
results = model(img)

for r in results:
    i = 1
    
    for box in r.boxes:
        if (box.cls==16):
            label = r.names[int(box.cls)]
            x,y,x2,y2 = int(box.xyxy[0][0]),int(box.xyxy[0][1]),int(box.xyxy[0][2]),int(box.xyxy[0][3])
            cv2.rectangle(img,(x,y),(x2,y2),(0,255,0),2)
            cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0))
        i+=1
    #r.save(filename="sonuc.jpg")
    cv2.imshow("Sonuc", img)
    cv2.imwrite("sonuc.jpg", img)
#%%
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
model = YOLO("yolo11n.pt")


img = cv2.imread("Data/cat_dogs.jpeg")
results = model(img)

for r in results:
    dog_count= 0
    annotator = Annotator(img)
    
    for box in r.boxes:
        if (box.cls==16):
            dog_count+=1
    #r.save(filename="sonuc.jpg")
    
    label = str(dog_count) + " kopek var...!"
    cv2.putText(img, label, (40,143), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))
    
    cv2.imshow("Sonuc", img)
    cv2.imwrite("sonuc.jpg", img)









