from ultralytics import YOLO

model = YOLO("runs\\best_640_ppn\\weights\\best.pt")

model.val(data="ppn_zero_shot_0_1_val.yaml", batch=9)