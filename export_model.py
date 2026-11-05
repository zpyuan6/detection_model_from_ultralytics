from ultralytics import YOLO
import yaml
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import time
import torch
from torchvision import transforms
import cv2

def export_ptl(model_path):
    model = YOLO(model_path)
    model.fuse()
    model.info(verbose=True)
    model.export(format='torchscript', imgsz=640)

    # model.export(format='torchscript', dynamic=True, optimize=True)

# def run_ptl(model):

def export_ncnn(model_path):
    model = YOLO(model_path)
    # model.export(format='onnx', imgsz=640, half=True, opset = 11)
    # model.export(format='onnx', imgsz=640, half=True)
    # model.export(format='onnx', imgsz=640)

    # model.export(format='onnx', imgsz=640, half=True, opset = 12)
    # model.export(format='onnx', imgsz=640, half=True, opset = 11)
    # model.export(format='onnx', imgsz=640, half=True, opset = 13)
    # model.export(format='ncnn', half=True, imgsz=640)
    # model.export(format='ncnn', half=True, imgsz=640, opset = 13)
    # model.export(format='ncnn', imgsz=640)
    model.export(format='ncnn', imgsz=640, opset = 11)


def export_tflite(model_path):
    model = YOLO(model_path)
    model.export(format='mlmodel', int8=True, imgsz=320)


def load_image_and_labels(image_path: str, label_path: str):
    """Load an image and its corresponding labels."""
    image = Image.open(image_path).convert("RGB")

    pixel_width, pixel_height = image.size

    with open(label_path, 'r') as file:
        raw_labels = [line.strip().split() for line in file.readlines()]

        labels = []

        for raw_label in raw_labels:
            if len(raw_label) == 5:
                category, cx, cy, w, h = raw_label
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                labels.append({
                    'category_id': int(category),
                    'bbox': [(cx - 0.5*w) * pixel_width, (cy - 0.5*h) * pixel_height, (cx + 0.5*w) * pixel_width, (cy + 0.5*h) * pixel_height]
                    })


    return image, labels


def compute_iou(pred_box1, anno_box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = pred_box1
    x1g, y1g, x2g, y2g = anno_box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def val_exported_model(model_path, val_annotations_file, imgsz):
    model = YOLO(model_path)

    model.val(data=val_annotations_file, imgsz=imgsz)

    # with open(val_annotations_file, 'r') as f:
    #     dataset_info = yaml.safe_load(f)

    # image_dir = dataset_info['val']
    # label_dir = image_dir.replace('images', 'labels')

    # image_files = [f for f in os.listdir(os.path.join(dataset_info['path'], image_dir)) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG') or f.endswith('.jpeg') or f.endswith('.JPEG')]

    # all_results = []
    # all_labels = []
    # all_scores = []
    # all_ious = []

    # t = 0
    # for image_file in tqdm(image_files):
    #     image_path = os.path.join(dataset_info['path'], image_dir, image_file)
    #     file_name = ".".join(image_file.split(".")[:-1])
    #     label_path = os.path.join(dataset_info['path'], label_dir, f"{file_name}.txt")

    #     _, annos = load_image_and_labels(image_path, label_path)
        
    #     start_time = time.time()
    #     results = model(image_path, imgsz=imgsz)[0].boxes
    #     end_time = time.time()
    #     t += (end_time - start_time)

    #     boxes, confs, label = results.xyxy, results.conf, results.cls

    #     for box, score, label in zip(boxes, confs, label):
    #         all_results.append({
    #             'image_id': image_file,
    #             'category_id': label,
    #             'bbox': box,
    #             'score': score
    #         })

    #         all_scores.append(score)
    #         all_labels.append(label)

    #         ious = [compute_iou(box, l['bbox']) for l in annos if l['category_id'] == label]

    #         all_ious.append(max(ious) if ious else 0)

    # categories = dataset_info['names']

    # detection_speed = t / len(image_files)

    # aps_per_class = []
    # for i, category in categories.items():
    #     y_true = [1 if iou >= 0.5 and label == i else 0 for iou, label in zip(all_ious, all_labels)]
    #     if sum(y_true) == 0:
    #         ap = 0
    #     else:
    #         ap = average_precision_score(y_true, all_scores)
    #     print(f"AP for class {category}: {ap}")
    #     aps_per_class.append(ap)

    # mAP = sum(aps_per_class) / len(aps_per_class)
    # print(f"mAP: {mAP}")
    # print(f"Detection speed: {1/detection_speed} FPS")




def save_mmodel_as_torchscript(model_path):
    model = YOLO(model_path)
    model.fuse()
    model.info(verbose=True)
    model.export(format='torchscript', imgsz=320)

def letter_box(image, target_size):
    ih, iw = image.shape[:2]
    w, h = target_size

    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    img = cv2.resize(image, (nw, nh))

    dw = (w - nw) // 2
    dh = (h - nh) // 2

    top, bottom = dh, h - nh - dh
    left, right = dw, w - nw - dw
    img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return img, scale, (dw, dh)


def test_yolov10(model_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])


    model = torch.jit.load(model_path)

    img = cv2.imread("F:\\nematoda\\AgriNema\\Formated_Dataset\\Yolo_11Dec\\images\\train\\cMC170001 (95).JPG")
    input_img, scale, (dw, dh) = letter_box(img, (320,320))
    x = input_img.transpose(2, 0, 1) / 255
    x = torch.tensor(x).unsqueeze(0).float()

    predict = model(x)

    threshold = 0.4

    filter_results = predict[predict[:,:,4] > threshold]

    xyxy = filter_results[:, :4]
    xyxy[:,[0,2]] -= dw
    xyxy[:,[1,3]] -= dh
    xyxy /= scale
    conf = filter_results[:, 4:5].squeeze()
    cls = filter_results[:, 5:].squeeze()

    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(xyxy.tolist(), cls.tolist(), conf.tolist()):
        # 绘制矩形
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        # 绘制文本
        text = f"{label} {score:.2f}"
        cv2.putText(img, text, (int(box[0]), int(box[1] - 10)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == "__main__":
    # model_path = "runs\\detect\\our_medium_all\\weights\\best.pt"
    # export_ptl(model_path)

    model_path = "runs\\detect\\yolov10n_320_ppn2\\weights\\best.pt"

    # export_ptl(model_path)

    # export_ncnn(model_path)

    # export_tflite(model_path)

    # save_mmodel_as_torchscript(model_path)

    # val_exported_model("runs\\yolov8_320_uk_pest_dataset_07JAN25_all_insect_X2_tiny\\weights\\best_saved_model\\best_int8.tflite", "uk_pest_dataset_07JAN25_all_insect_x2.yaml", 320)

    test_yolov10("runs\\detect\\yolov10n_320_ppn2\\weights\\best.torchscript")