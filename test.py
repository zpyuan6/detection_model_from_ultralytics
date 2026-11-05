import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
# sample = np.ones([128,128,3])

# plt.imshow(sample)
# plt.show()
# model = YOLO('runs\\uk_pest_01JAN_tiny\\args.yaml')
model = YOLO('runs\\detect\\yolov8n_320_aphid_4k\\weights\\best.pt')
# model._load('runs\\uk_pest_01JAN_tiny\\weights\\best.pt')
# model = YOLO()
# 0x1c5c1c5d1d3d1d11
# 0x1bff6fec30311
results = model('C:\\Datasets\\APHID-4K-YOLO\\images\\val\\001_012845.jpg')

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')

