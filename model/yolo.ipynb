# install  ultralytics
!pip install -U ultralytics

# rest of the code 
from IPython.display import Image, display
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
dataset_yaml = """
names:
- elbow positive
- fingers positive
- forearm fracture
- humerus
- humerus fracture
- shoulder fracture
- wrist positive

nc: 7

train: Bone-Fracture-Detection-3/train/images
val: Bone-Fracture-Detection-3/valid/images
test: Bone-Fracture-Detection-3/test/images
"""
with open("data.yaml", "w") as f:
    f.write(dataset_yaml)



model.train(data="/content/Bone-Fracture-Detection-3/data.yaml", epochs=100, batch=16, imgsz=416, amp=True)
metrics = model.val()
results = model.predict(source='/content/Bone-Fracture-Detection-3/test/images', save=True)
model.save('/content/drive/MyDrive/best_bone_fracture_model.pt')
results = model.val(data="/content/drive/MyDrive/archive (4)/BoneFractureYolo8/data.yaml")
print("Precision:", results.box.p[0])
print("Recall:", results.box.r[0])
print("F1 Score:", results.box.f1[0])
print("mAP50:", results.maps[0])
print("mAP50-95:", results.maps[1])
