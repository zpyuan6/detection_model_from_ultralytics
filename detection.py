from ultralytics import YOLO
import cv2
import time
import os

# classification = ["Meloidogyne", "Cyst", "Globodera", "Pratylenchus", "PCN j2", "Ditylenchus"]



def prediction_test():
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model("bus.jpg")  # predict on an image

    res_plotted = results[0].plot()

    print("Results class", results[0])
    #  format x1,y1,x2,y2,conf,cls
    print("Bounding box", results[0].boxes)
    print("Bounding box", results[0].boxes.xyxy)

    # print(results[0].masks)
    # print(results[0].probs)

    # print(type(res_plotted))
    cv2.imshow("result", res_plotted)
    cv2.waitKey()
    # print(results[0].boxes)

def prediction(model_path,image_path, conf=0.5, savepath=None):
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    t1 = time.time()
    results = model(image_path, imgsz=320, conf=conf)  # predict on an image
    t2 = time.time()
    print(f"Spend {(t2-t1)*1000} ms ")
    print(results[0].boxes)

    # #  format x1,y1,x2,y2,conf,cls
    # print("Bounding box", results[0])
    # print("Bounding box", results[0].boxes.xyxy)

    # print("Results class", results[0].names[int(results[0].boxes.cpu().data[0][-1].item())], )

    img = cv2.imread(image_path)
    cv2.namedWindow("result",0)

    result_static = {}

    for i, bounding_box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        if results[0].boxes.cls.cpu().numpy()[i]==0:
            color = (0,255,0)
        elif results[0].boxes.cls.cpu().numpy()[i]==1:
            color = (0,0,255)
        elif results[0].boxes.cls.cpu().numpy()[i]==2:
            color = (126,0,0)
        elif results[0].boxes.cls.cpu().numpy()[i]==3:
            color = (255,0,0)
        elif results[0].boxes.cls.cpu().numpy()[i]==4:
            color = (255,255,0)
        elif results[0].boxes.cls.cpu().numpy()[i]==5:
            color = (255,0,255)
        else:
            color = (255,0,255)

        # print(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
        img = cv2.rectangle(img, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]),int(bounding_box[3])),color,10)

        item_name = results[0].names[int(results[0].boxes.cpu().data[i][-1].item())]
        # img = cv2.putText(img, item_name, (int(bounding_box[0]),int(bounding_box[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 10)

        if item_name in result_static:
            result_static[item_name] += 1
        else: 
            result_static[item_name] = 1

    # num = len(results[0].boxes.xyxy.cpu())
    # class_result = results[0].names[int(results[0].boxes.cpu().data[i][-1].item())]
    # img = cv2.putText(img, f"; ".join([f"{k}: {v}" for k,v in result_static.items()]), (10,100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (64,224,208), 10)
    print(result_static)
    cv2.imshow("result", img)
    cv2.waitKey()
    img_name = ".".join(image_path.split("\\")[-1].split(".")[:-1]) 

    if savepath!=None:
        cv2.imwrite(savepath,img)
    else:
        cv2.imwrite(f"{img_name}_result.png",img)

def is_image_file(file_name):
    if file_name.split(".")[-1] == "JPG" or file_name.split(".")[-1] == "jpg" or file_name.split(".")[-1] == "JPEG" or file_name.split(".")[-1] == "tif":
        return True
    return False

def draw_results_from_folder(model_path, folder_path):
    save_path = os.path.join(folder_path, "detection_result")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for root,folder,files in os.walk(folder_path):
        if root!=folder_path:
            break
        for file in files:
            if is_image_file(file):
                prediction(model_path, os.path.join(root, file), 0.585, os.path.join(save_path,file))
                


# Yolov8 DOCS https://docs.ultralytics.com/modes/predict/
if __name__ == "__main__":
    # prediction_test()
    # prediction("runs\\detect\\train\\weights\\best.pt", "F:\\nematoda\\nemadote_detection\\images\\train\\img_5010_id246.jpg")

    prediction("runs\\nematodes_medium_21_10\\weights\\best.pt", "C:\\Users\\zhipeng\\Desktop\\Pratylenchus spp2\\Image001.jpg")

    # pathes = [
    #     "F:\\nematoda\\AgriNema\\unannotated_data\\PCN_RLN_x10",
    #     # "F:\\nematoda\\AgriNema\\unannotated_data\\Test\\annotation",
    #     # "F:\\nematoda\\AgriNema\\unannotated_data\\Potato cyst nematodes (for AI training)",
    #     # "F:\\nematoda\\AgriNema\\unannotated_data\\PCN+RLN_Sept",
    #     # "F:\\nematoda\\AgriNema\\unannotated_data\\PCN_RLN_x5"
    #     ]

    # for path in pathes:
    #     draw_results_from_folder("runs\\nematodes_tiny_21_10\\weights\\best.pt", path)


    # prediction("runs\\yolov8_320_uk_pest_dataset_07JAN25_all_insect_X2_tiny\\weights\\best_saved_model\\best_int8.tflite", "X:\\pervasive_group\\PestProject\\YOLO_07JAN25_ALL_INSECT_ZOOM_IN_2X\\images\\train\\001_000962.jpg")