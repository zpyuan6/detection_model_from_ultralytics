import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def crop_image_and_annotation(orginal_yolo_path, output_dir, scale=2.0, data_type='val'):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(orginal_yolo_path, "images", data_type)
    label_path = os.path.join(orginal_yolo_path, "labels", data_type)

    target_image_path = os.path.join(output_dir, "images", data_type)
    target_label_path = os.path.join(output_dir, "labels", data_type)
    os.makedirs(target_image_path, exist_ok=True)
    os.makedirs(target_label_path, exist_ok=True)


    # 处理每一张图片和对应的标注
    for filename in os.listdir(image_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            image_file = os.path.join(image_path, filename)
            label_file = os.path.join(label_path, filename.replace(".JPG", ".txt").replace(".jpg", ".txt"))
            
            # 读取图片
            img = Image.open(image_file)
            width, height = img.size

            target_width = int(width / scale)
            target_height = int(height / scale)

            # 处理对应的标注
            if os.path.exists(label_file):
                with open(label_file, 'r') as file:
                    lines = file.readlines()

                # 初始化最小和最大边界
                min_x, min_y, max_x, max_y = width, height, 0, 0

                object_cluster_center_x_sum = 0
                object_cluster_center_y_sum = 0
                object_count = 0

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = parts[0]
                    x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts[1:])
                    
                    # 计算绝对坐标
                    x_center = x_center_rel * width
                    y_center = y_center_rel * height
                    obj_width = width_rel * width
                    obj_height = height_rel * height

                    # 更新最小最大边界
                    min_x = min(min_x, x_center - obj_width / 2)
                    min_y = min(min_y, y_center - obj_height / 2)
                    max_x = max(max_x, x_center + obj_width / 2)
                    max_y = max(max_y, y_center + obj_height / 2)

                    object_cluster_center_x_sum += x_center
                    object_cluster_center_y_sum += y_center
                    object_count += 1
                

                object_cluster_center_x = object_cluster_center_x_sum / object_count
                object_cluster_center_y = object_cluster_center_y_sum / object_count 

                # 计算扩展后的边界
                if target_width < (max_x - min_x):
                    crop_x1 = max(0, int(min_x))
                    crop_x2 = min(width, int(max_x))
                else:
                    crop_x1 = max(0, (min_x+max_x)/2 - target_width/2)  
                    crop_x2 = min(width, crop_x1 + target_width)
                    if crop_x2 == width:
                        crop_x1 = width - target_width

                if target_height < (max_y - min_y):
                    crop_y1 = max(0, int(min_y))
                    crop_y2 = min(height, int(max_y))
                else:
                    crop_y1 = max(0, (min_y+max_y)/2 - target_height/2)
                    crop_y2 = min(height, crop_y1 + target_height)
                    if crop_y2 == height:
                        crop_y1 = height - target_height


                # 裁剪并保存新图像
                cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                new_image_file = os.path.join(target_image_path, filename)
                cropped_img.save(new_image_file)

                # 更新并保存标注
                new_label_content = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = parts[0]
                    x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts[1:])
                    x_center = x_center_rel * width
                    y_center = y_center_rel * height
                    obj_width = width_rel * width
                    obj_height = height_rel * height

                    new_x_center = (x_center - crop_x1) / (crop_x2 - crop_x1)
                    new_y_center = (y_center - crop_y1) / (crop_y2 - crop_y1)
                    new_width = obj_width / (crop_x2 - crop_x1)
                    new_height = obj_height / (crop_y2 - crop_y1)

                    if new_x_center < 0 or new_x_center > 1 or new_y_center < 0 or new_y_center > 1:
                        print(f"Invalid coordinate: {x_center}, {y_center}")

                    new_label_content.append(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")

                new_label_file = os.path.join(target_label_path, filename.replace(".jpg", ".txt").replace(".JPG", ".txt"))
                with open(new_label_file, 'w') as file:
                    file.writelines(new_label_content)

def draw_zoom_results(output_dir, data_type='val'):
    image_path = os.path.join(output_dir, "images", data_type)
    label_path = os.path.join(output_dir, "labels", data_type)

    for file in os.listdir(image_path):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            image_file = os.path.join(image_path, file)
            label_file = os.path.join(label_path, file.replace(".JPG", ".txt").replace(".jpg", ".txt"))

            with Image.open(image_file) as img:
                draw = ImageDraw.Draw(img)
                width, height = img.size

                # 检查标注文件是否存在
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        class_id = parts[0]
                        x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts[1:])

                        # 计算绝对坐标
                        x_center = x_center_rel * width
                        y_center = y_center_rel * height
                        box_width = width_rel * width
                        box_height = height_rel * height

                        # 计算边界框的左上角和右下角
                        x1 = x_center - box_width / 2
                        y1 = y_center - box_height / 2
                        x2 = x_center + box_width / 2
                        y2 = y_center + box_height / 2

                        # 绘制边界框
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # 显示图像
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.title(f"Annotated Image: {file}")
                plt.axis('off')
                plt.show()

if __name__ == "__main__":
    orginal_yolo_path = 'C:\\Datasets\\APHID-4K-YOLO'
    output_dir = 'C:\\Datasets\\APHID-4K-YOLO_ZOOM_IN_2X'
    crop_image_and_annotation(orginal_yolo_path, output_dir, 2, 'train')
    crop_image_and_annotation(orginal_yolo_path, output_dir, 2, 'val')
    # draw_zoom_results(output_dir)

    # output_dir = 'X:\\pervasive_group\\PestProject\\Dataset\\YOLO_18SEP24_ALL_INSECTA_ZOOM_IN_3X'
    # crop_image_and_annotation(orginal_yolo_path, output_dir, 3, 'train')
    # crop_image_and_annotation(orginal_yolo_path, output_dir, 3, 'val')
    # # draw_zoom_results(output_dir)

    # output_dir = 'X:\\pervasive_group\\PestProject\\Dataset\\YOLO_18SEP24_ALL_INSECTA_ZOOM_IN_4X'
    # crop_image_and_annotation(orginal_yolo_path, output_dir, 4, 'train')
    # crop_image_and_annotation(orginal_yolo_path, output_dir, 4, 'val')
    # # draw_zoom_results(output_dir)