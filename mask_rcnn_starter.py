# coding: utf-8

# Algorytm Mask R-CNN 
# Użyto wytrenowanego modelu COCO

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Katalog root 
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)  
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Konfiguracja COCO 
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# Folder na logi i wytrenowany model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Plik z wytrenowanymi wagami
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Pobierz wagi, jeśli ich nie ma
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Folder na obrazki
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# KONFIGURACJA

# Model wytrenowany na zbiorze MS-COCO. Konfiguracja tego modelu jest w klasie ```CocoConfig``` w ```coco.py```.

class InferenceConfig(coco.CocoConfig):
    # Jeden obraz przetwarzany w danej chwili
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Utwórz model i załaduj wagi

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("Model utworzony!")

model.load_weights(COCO_MODEL_PATH, by_name=True)
print("Wagi z COCO załadowane")


# Nazwy klas
# 
# Model zwraca ID klasy (liczby całkowite), jednoznacznie ją identyfikując. Np. w COCO klasa 'person' ma ID=1, a klasa 'teddy bear' ma ID=88. W COCO numery klas nie są sekwencyjne, co oznacza że mogą występować "przeskoki", np. z 70 na 72 (brak ID=71)71.

# Użycie: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Detekcja obiektów

# Załaduj losowy obrazek
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Uruchom detekcję
results = model.detect([image], verbose=1)

# Wizualizacja (najlepiej działa za pomocą Jupyter Notebook)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

