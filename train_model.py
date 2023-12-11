from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Yolov8 DOCS https://docs.ultralytics.com/modes/train/
if __name__ == "__main__":
    # model = YOLO("yolov8m.pt")
    # model.train(data="nematodes.yaml", epochs=100, imgsz=640)
    model_list = ["yolov8n.pt","yolov8m.pt","yolov8x.pt"]
    # model_list = ["/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_tiny_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_medium_07_06/weights/best.pt","/home/zhipeng/Desktop/nametodes/Yolov8_for_nematodes/runs/detect/pest_uk_extra_07_06/weights/best.pt"]

    model_large = YOLO(model_list[0])
    model_large.train(data="nematodes.yaml", epochs=400, imgsz=640, batch=32, name= "nematodes_tiny_31_08")

    model_large = YOLO(model_list[1])
    model_large.train(data="nematodes.yaml", epochs=200, imgsz=640, batch=16, name= "nematodes_medium_31_08")

    model_large = YOLO(model_list[2])
    model_large.train(data="nematodes.yaml", epochs=200, imgsz=640, batch=2, name= "nematodes_extra_31_08")
