import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def image_number(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num+=1

    print(num)
            
def count_object_from_yolo_label_folder(path):
    num = 0
    object_stat = {}
    file_stat = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                num += len(lines)
                if len(lines[0])>1:
                    class_id = int(lines[0].split(" ")[0])
                    if class_id in file_stat:
                        file_stat[class_id] += 1
                    else:
                        file_stat[class_id] = 1

                for line in lines:
                    if len(line)>1:
                        class_id = int(line.split(" ")[0])
                        if class_id in object_stat:
                            object_stat[class_id] += 1
                        else:
                            object_stat[class_id] = 1
    print(num)
    print(object_stat)
    print(file_stat)

def calculate_object_area(path):
    num = 0
    object_area = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                num += len(lines)
                for line in lines:
                    arr = line.split(" ")
                    w,h  = arr[3],arr[4]
                    object_area += float(w)*float(h)

    print(object_area)   
    print(num)
    print(object_area/num)   


def count_object_from_voc_label_folder(path):
    class_list = set([])
    sample_number = {}
    object_number = {}

    samples_for_each_class = {}
    for root, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "xml":
                print(file)
                voc_annotation_file = open(os.path.join(root,file), encoding='utf-8')
                tree = ET.parse(voc_annotation_file)
                r = tree.getroot()

                temp_list = set([])

                for obj in r.iter('object'):
                    # cls_id = 0
                    # xmlbox = obj.find('bndbox')
                    # b = [int(xmlbox.find('xmin').text)-50, int(xmlbox.find('ymin').text)-50, int(xmlbox.find('xmax').text)+50, int(xmlbox.find('ymax').text)+50]
                    # bounding_boxes.append(b)
                    class_name = obj.find('name').text
                    class_list.add(class_name)
                    temp_list.add(class_name)
                    if class_name in object_number:
                        object_number[class_name] = object_number[class_name]+1
                    else:
                        object_number[class_name] = 1

                for class_name in temp_list:
                    if class_name in sample_number:
                        sample_number[class_name] = sample_number[class_name]+1
                    else:
                        sample_number[class_name] = 1

                    if class_name in samples_for_each_class and len(samples_for_each_class[class_name])<10:
                        samples_for_each_class[class_name].append(file.split(".")[0])
                    else:
                        samples_for_each_class[class_name]= [file.split(".")[0]]

    # print(samples_for_each_class)


    print(class_list,sample_number,object_number)


def select_sample(path, target_path):
    class_list = set([])
    sample_number = {}
    object_number = {}

    samples_for_each_class = {}
    for root, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == "xml":
                print(file)
                voc_annotation_file = open(os.path.join(root,file), encoding='utf-8')
                tree = ET.parse(voc_annotation_file)
                r = tree.getroot()

                temp_list = set([])

                for obj in r.iter('object'):
                    # cls_id = 0
                    # xmlbox = obj.find('bndbox')
                    # b = [int(xmlbox.find('xmin').text)-50, int(xmlbox.find('ymin').text)-50, int(xmlbox.find('xmax').text)+50, int(xmlbox.find('ymax').text)+50]
                    # bounding_boxes.append(b)
                    class_name = obj.find('name').text
                    class_list.add(class_name)
                    temp_list.add(class_name)
                    if class_name in object_number:
                        object_number[class_name] = object_number[class_name]+1
                    else:
                        object_number[class_name] = 1

                for class_name in temp_list:
                    if class_name in sample_number:
                        sample_number[class_name] = sample_number[class_name]+1
                    else:
                        sample_number[class_name] = 1

                    if class_name in samples_for_each_class:
                        print(len(samples_for_each_class[class_name]))

                    if class_name in samples_for_each_class and len(samples_for_each_class[class_name])<20:
                        samples_for_each_class[class_name].add(file.split(".")[0])
                    else:
                        samples_for_each_class[class_name]= set([file.split(".")[0]])

    print(samples_for_each_class)

    for key in samples_for_each_class.keys():
        for file_id in samples_for_each_class[key]:
            shutil.copy2(os.path.join(path, file_id+".xml"), os.path.join(target_path, file_id+".xml"))
            shutil.copy2(os.path.join(path, file_id+".JPG"), os.path.join(target_path, file_id+".JPG"))


def statistic_object_size(yolo_dataset_path):
    class_area_ratios = {}

    for root, folders, files in os.walk(yolo_dataset_path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])
                    if width>=1 or height>=1:
                        continue
                    area_ratio = width*height
                    
                    if class_id not in class_area_ratios:
                        class_area_ratios[class_id] = []
                    class_area_ratios[class_id].append(area_ratio)

    stats_info = {}

    for class_id, ratios in class_area_ratios.items():
        ratios = np.array(ratios)
        mean_val = np.mean(ratios)
        max_val = np.max(ratios)
        min_val = np.min(ratios)
        var_val = np.var(ratios)
        median_val = np.median(ratios)
        confidence_interval = stats.t.interval(0.95, len(ratios)-1, loc=mean_val, scale=stats.sem(ratios))

        stats_info[class_id] = confidence_interval
        # {
        #     "mean": mean_val,
        #     "max": max_val,
        #     "min": min_val,
        #     "variance": var_val,
        #     "median": median_val,
        #     "confidence_interval": confidence_interval
        # }

    print(stats_info)

def draw_object_size_distribution(yolo_dataset_path):
    class_area_ratios = []

    for root, folders, files in os.walk(yolo_dataset_path):
        for file in files:
            if file.split(".")[-1] == "txt":
                f = open(os.path.join(root,file),"r")
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])
                    if width>=1 or height>=1:
                        continue
                    area_ratio = width*height
                    
                    class_area_ratios.append({
                        "class_id": class_id,
                        "width": width,
                        "height": height,
                        "area_ratio": area_ratio
                    })

    df = pd.DataFrame(class_area_ratios)

    # print(df)

    df = df[df['area_ratio'] < 0.01]

    print(max(df['area_ratio']), min(df['area_ratio']))

    plt.figure(figsize=(10, 6))
    
    # 使用Seaborn绘制面积分布
    sns.histplot(data = df, x = 'area_ratio', kde=True, hue='class_id', palette='tab10', bins=30)
    
    # plt.title("Distribution of Target Sizes (Area Ratio) by Class")
    # plt.xlabel("Area Ratio (Normalized)")
    # plt.ylabel("Frequency")
    # plt.legend(title="Class ID", loc="upper right")
    # plt.tight_layout()
    plt.show()

    



if __name__ == "__main__":
    # image_number("F:\\nematoda\\nema")
    # count_object_from_yolo_label_folder("F:\\nematoda\\AgriNema\\Formated_Dataset\\Yolo_0831\\labels")
    # calculate_object_area("F:\\nematoda\\nemadote_detection\\labels")

    path = "F:\\pest_data\\Multitask_or_multimodality\\annotated_data"

    count_object_from_voc_label_folder("F:\\pest_data\\Aphids_VOC2007\\Annotations")
    # select_sample(path, "C:\\Users\\zhipeng\\Desktop\\DataSample")

    # statistic_object_size("F:\\pest_data\\Multitask_or_multimodality\\YOLO_25JUN24_ALL_INSECT\\labels\\train")

    # draw_object_size_distribution("F:\\pest_data\\Multitask_or_multimodality\\YOLO_18SEP24_ALL_INSECTA\\labels\\train")