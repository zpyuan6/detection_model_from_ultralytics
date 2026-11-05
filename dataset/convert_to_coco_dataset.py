import json
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 
import random
from datetime import datetime
import shutil
from pathlib import Path

def convert_voc_to_coco(voc_folder, coco_folder, version=1.0, split_ratio=0.9):
    """
    Note: This function is specific for converting VOC format to COCO format. Renaming categories is not implemented in this function.
    """
    coco_folder = os.path.join(coco_folder, f"coco_pest_{datetime.now().date().strftime('%Y_%m_%d')}")

    image_folder = os.path.join(coco_folder, 'images')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    data_dict = {}
    data_dict['info'] = {
        "description": "UK Pest dataset",
        "version": version,
        "year": datetime.now().year,
        "date_created": datetime.now().date().strftime('%Y-%m-%d')
    }
    category_set = set()
    
    xml_files = [f for f in listdir(voc_folder) if f.endswith('.xml') ]

    images_list = []
    annotations_list = []
    image_id = 0
    annotation_id = 0
    
    with tqdm(total=len(xml_files)) as pbar:
        for xml_file in xml_files:
            tree = ET.parse(join(voc_folder, xml_file))
            root = tree.getroot()
            file_name = ".".join(xml_file.split('.')[0:-1])  + '.JPG'
            # if not isfile(join(voc_folder, file_name)):
            #     file_name = xml_file.split('.')[0] + '.JPG'
            if not isfile(join(voc_folder, file_name)):
                file_name = ".".join(xml_file.split('.')[0:-1]) + '.jpg'
                if not isfile(join(voc_folder, file_name)):
                    print(f'File not found: {file_name}')
                    continue

            shutil.copy2(os.path.join(voc_folder, file_name), os.path.join(image_folder, file_name))

            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            images_list.append({
                    'file_name': file_name,
                    'height': height,
                    'width': width,
                    'id': image_id
            })

            for obj in root.findall('object'):
                category = obj.find('name').text
                category_set.add(category)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                annotations_list.append({
                    'area': (xmax - xmin) * (ymax - ymin),
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'category_id': list(category_set).index(category) + 1,
                    'id': annotation_id,
                    'image_id': image_id,
                    'iscrowd': 0,
                    'segmentation': []
                })
                annotation_id += 1
                
            image_id += 1

            pbar.update(1)


    train_images = random.sample(images_list, int(len(images_list)*split_ratio))
    train_image_ids = [image['id'] for image in train_images]
    train_annotations = [annotation for annotation in annotations_list if annotation['image_id'] in train_image_ids]

    val_images = [item for item in images_list if item['id'] not in train_image_ids]
    val_image_ids = [image['id'] for image in val_images]
    val_annotations = [annotation for annotation in annotations_list if annotation['image_id'] in val_image_ids]

    print(f"Number of images: {len(images_list)}; Number of train images: {len(train_images)}; Number of val images: {len(val_images)}")

    categories_list = [{'id': i + 1, 'name': cat, 'supercategory': 'none'} for i, cat in enumerate(category_set)]
    
    train_dict = {**data_dict, 'images': train_images, 'annotations': train_annotations, 'categories': categories_list}
    val_dict = {**data_dict, 'images': val_images, 'annotations': val_annotations, 'categories': categories_list}

    coco_annotation_folder = os.path.join(coco_folder, 'annotations')

    if not os.path.exists(coco_annotation_folder):
        os.makedirs(coco_annotation_folder)
    
    with open(os.path.join(coco_annotation_folder, "trainPEST.json"), 'w') as f:
        json.dump(train_dict, f, indent=4)
    with open(os.path.join(coco_annotation_folder, "valPEST.json"), 'w') as f:
        json.dump(val_dict, f, indent=4)

    return coco_folder

def rename_categories_for_coco(coco_folder, original_annotation_file, target_annotation_file, rename_mapping:dict):

    original_annotation_path = os.path.join(coco_folder, "annotations", original_annotation_file)

    target_annotation_path = os.path.join(coco_folder, "annotations", target_annotation_file)

    with open(original_annotation_path, 'r') as f:
        original_annotation = json.load(f)

    categories_list = []
    new_categories = []
    category_id_map = {}

    for category in original_annotation['categories']:
        old_name = category['name']
        if old_name in rename_mapping:
            new_name = rename_mapping[old_name]
        else:
            new_name = old_name

        if new_name not in categories_list:
            categories_list.append(new_name)
            new_categories.append({
                'id': categories_list.index(new_name)+1,
                'name': new_name,
                'supercategory': 'none'
            })
        

        if category['id']!=categories_list.index(new_name)+1:
            category_id_map[category['id']] = categories_list.index(new_name)+1


        print(f"Renamed {old_name} to {new_name}")
        print(f"Category ID: {category['id']} -> {categories_list.index(new_name)+1}")

    print(f"category_id_map: {category_id_map}")

    for annotation in original_annotation['annotations']:
        if annotation['category_id'] in category_id_map:
            annotation['category_id'] = category_id_map[annotation['category_id']]

    original_annotation['categories'] = new_categories

    with open(target_annotation_path, 'w') as f:
        json.dump(original_annotation, f, indent=4)
    

def update_supercategories(coco_folder, annotation_file, supercategories_mapping:dict):
    """
    supercategories_mapping: {category_name: supercategory_name}
    """
    with open(os.path.join(coco_folder, "annotations", annotation_file), 'r') as f:
        data = json.load(f)

    updated = False
    for category in data['categories']:
        if category['name'] in supercategories_mapping:
            category['supercategory'] = supercategories_mapping[category['name']]
            updated = True

    if updated:
        with open(os.path.join(coco_folder, "annotations", annotation_file), 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print('No categories updated')


def unified_image_format(coco_folder):
    target_format='JPG'

    image_folder = os.path.join(coco_folder, 'images')
    for image_file in os.listdir(image_folder):
        if image_file.split('.')[-1]!= target_format:
            image_path = os.path.join(image_folder, image_file)
            new_image_path = os.path.join(image_folder, image_file.split('.')[0] + '.' + target_format)
            os.rename(image_path, new_image_path)
            print(f"Renamed {image_path} to {new_image_path}")

    for annotation_file in os.listdir(os.path.join(coco_folder, 'annotations')):
        if annotation_file.endswith('.json'):
            with open(os.path.join(coco_folder, 'annotations', annotation_file), 'r') as f:
                data = json.load(f)

            for image in data['images']:
                if image['file_name'].split('.')[-1] != target_format:
                    image['file_name'] = image['file_name'].split('.')[0] + '.' + target_format

            with open(os.path.join(coco_folder, 'annotations', annotation_file), 'w') as f:
                json.dump(data, f, indent=4)


def unified_image_format_annotation(voc_folder):
    target_format='JPG'
    original_format='jpg'

    image_folder = os.path.join(voc_folder)
    for image_file in os.listdir(image_folder):
        if image_file.split('.')[-1] == original_format:
            image_path = os.path.join(image_folder, image_file)
            new_image_name = image_file.split('.')[0] + '.' + target_format
            new_image_path = os.path.join(image_folder, new_image_name)
            if new_image_name in os.listdir(image_folder):
                raise Exception(f"File already exists: {new_image_path}")

            os.rename(image_path, new_image_path)
            print(f"Renamed {image_path} to {new_image_path}")


def check_coco_folder(coco_folder):
    image_list = os.listdir(os.path.join(coco_folder, 'images'))

    for annotation_file in os.listdir(os.path.join(coco_folder, 'annotations')):
        if annotation_file.endswith('.json'):
            with open(os.path.join(coco_folder, 'annotations', annotation_file), 'r') as f:
                data = json.load(f)

            print(f"Checking {annotation_file} length: {len(data['images'])}")
            for image in data['images']:
                if image['file_name'] not in image_list:
                    print(f"File not found: {image['file_name']}")

                if image['file_name'].split('.')[-1] != 'JPG':
                    print(f"Wrong format: {image['file_name']}")

    with open(os.path.join(coco_folder, 'annotations', 'trainPEST.json'), 'r') as f:
        data = json.load(f)
        train_list = [image['file_name'] for image in data['images']]
    
    with open(os.path.join(coco_folder, 'annotations', 'valPEST.json'), 'r') as f:
        data = json.load(f)
        val_list = [image['file_name'] for image in data['images']]
    for image in val_list:
        if image in train_list:
            print(f"Image in both train and val: {image}")
    

def check_image_file_in_VOC_and_COCO(voc_folder, coco_folder):
    voc_image_list = [file for file in os.listdir(voc_folder) if file.split('.')[-1]=='JPG']
    coco_image_list = os.listdir(os.path.join(coco_folder, 'images'))

    print(f"Number of images in VOC: {len(voc_image_list)}")
    print(f"Number of images in COCO: {len(coco_image_list)}")

    lossed_images = []

    for image in voc_image_list:
        if image not in coco_image_list:
            print(f"Image not found: {image}")
            lossed_images.append(image)

    if len(lossed_images)>0:
        print(f"Number of lossed images: {len(lossed_images)}")
        with open(os.path.join(coco_folder, 'annotations', 'trainPEST.json'), 'r') as f:
            data = json.load(f)
            train_list = [image['file_name'] for image in data['images']]
        
        with open(os.path.join(coco_folder, 'annotations', 'valPEST.json'), 'r') as f:
            data = json.load(f)
            val_list = [image['file_name'] for image in data['images']]

        for image in lossed_images:
            if image in train_list:
                print(f"Image in train: {image}")
            elif image in val_list:
                print(f"Image in val: {image}")
            else:
                print(f"Image not found in COCO: {image}")

def select_coco_samples_based_yolo(coco_path, yolo_path):
    yolo_image_list = [".".join(file.split(".")[0:-1]) for file in os.listdir(os.path.join(yolo_path, "images\\val"))]

    coco_dict = json.loads(open(os.path.join(coco_path, "annotations\\val.json")).read())

    coco_image_list = coco_dict['images']

    selected_images = []

    for image in coco_image_list:
        f_name = ".".join(image['file_name'].split(".")[0:-1])
        if f_name in yolo_image_list:
            selected_images.append(image)
        else:
            print(f"Image not found: {image['file_name']}")

    print(len(selected_images), len(yolo_image_list))

    coco_dict['images'] = selected_images

    b = json.dumps(coco_dict, indent=4)

    f2 = open(os.path.join(coco_path, "annotations\\selectedVal.json"), 'w')
    f2.write(b)
    f2.close()

def correct_coco_annotation_file(coco_folder, annotation_file):
    with open(os.path.join(coco_folder, "annotations", annotation_file), 'r') as f:
        data = json.load(f)

    old_categories = data['categories']

    new_categories = ["Meloidogyne", "Potato Cyst Nematode Cyst", "Globodera", "Pratylenchus", "Potato Cyst Nematode Juveniles 2", "Ditylenchus"]

    new_categories = [
        {
            "id": 0,
            "name": "Meloidogyne",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "Potato Cyst Nematode Cyst",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "Globodera",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "Pratylenchus",
            "supercategory": "none"
        },
        {
            "id": 4,
            "name": "Potato Cyst Nematode Juveniles 2",
            "supercategory": "none"
        },
        {
            "id": 5,
            "name": "Ditylenchus",
            "supercategory": "none"
        },
    ]

    # mapping = {
    #     11:1, 10:2, 2:3, 9:4, 7:5, 4:2, 8:6
    # }

    old_annotations = data['annotations']

    new_annotations = []

    for annotation in old_annotations:
        
        annotation['category_id'] = int(annotation['category_id'])+1
           

    # data['categories'] = new_categories
    
    with open(os.path.join(coco_folder, "annotations", annotation_file), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Example usage
    voc_folder = 'F:\\pest_data\\Multitask_or_multimodality\\annotated_data'
    coco_folder = 'F:\\pest_data\\Multitask_or_multimodality\\coco_pest_2024_10_01'
    # unified_image_format_annotation(voc_folder)

    # coco_folder = convert_voc_to_coco(voc_folder, "F:\\pest_data\\Multitask_or_multimodality")

    # coco_folder = convert_voc_to_coco("F:\\nematoda\\AgriNema\\original_annotated_data", "F:\\nematoda\\AgriNema\\Formated_Dataset\\COCO_Dec24", split_ratio=1.0)

    # coco_folder = convert_voc_to_coco("F:\\pest_data\\Multitask_or_multimodality\\annotated_data", "X:\\pervasive_group\\PestProject\\COCO_07JAN25_ALL_INSECT", split_ratio=1.0)

    select_coco_samples_based_yolo("X:\\pervasive_group\\PestProject\\COCO_07JAN25_ALL_INSECT\\coco_pest_2025_01_09", "X:\\pervasive_group\\PestProject\\YOLO_07JAN25_ALL_INSECT")

    # correct_coco_annotation_file("F:\\nematoda\\AgriNema\\Formated_Dataset\\COCO_Dec24", "selectedVal-copy.json")
    """
    classes_name_list_for_all_insecta = [
        "INSECTA", # 'INSECTA': 834, 'FROGHOPPER (CERCOPIDAE)': 1, 'SCARABAEIDAE': 1
        "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID", # 'GRAIN APHID (SITOBION AVENAE)', 'ROSE GRAIN APHID': 130+1, 
        "POLLEN BEETLE (MELIGETHES SPP.)", # 'POLLEN BEETLE (MELIGETHES SPP.)': 2843, 
        "SNAIL", # 'SNAIL': 50, 
        "CEREAL LEAF BEETLE (OULEMA MELANOPUS)", # 'CEREAL LEAF BEETLE (OULEMA MELANOPUS)': 66, 
        "FLY (DIPTERA)", # 'FLY (DIPTERA)': 1210, 'FLY (MELANOSTOMA SPP.)': 14, 'BIBIONID FLY (BIBIONIDAE)': 41, 'MUSCIDAE (FLY)': 4, 'LEAF MINERS': 2, 'BEAN SEED FLY (DELIA SPP.)': 49, 'FUNGUS GNAT (MYCETOPHILIDAE)': 8, 
        "CABBAGE STEM FLEA BETTLE", #'CABBAGE STEM FLEA BETTLE': 123, 
        "LADYBUG (COCCINELLIDAE)", #'LADYBUG (COCCINELLIDAE)': 198, 
        "LADYBUG (COCCINELLIDAE) (PUPA)", #'LADYBUG (COCCINELLIDAE) (PUPA)': 166, 
        "LADYBUG (COCCINELLIDAE) (LARVAE)", #'LADYBUG (COCCINELLIDAE) (LARVAE)': 10, 
        "SPIDER (ARANEUS SPP.)", #'SPIDER (ARANEUS SPP.)': 73, 
        "CHIRONOMID MIDGE", #'CHIRONOMID MIDGE': 93, 'CHIRONOMID MIDGE (MALE)': 4,
        "BEETLE (COLEOPTERA)", #'BEETLE (COLEOPTERA)': 176, 
        "MOSQUITO", # 'MOSQUITO': 77,
        "WASP", #'WASP': 36, 
        "SLUG", #'SLUG': 67, 
        "CABBAGE WHITEFLY", #'CABBAGE WHITEFLY': 13,
        "BEE", # 'BEE': 35,  
        "HEMIPTERA (PLANT BUG)", #'HEMIPTERA (PLANT BUG)': 4,
        "EARTHWORM", #'EARTHWORM': 7, 
        "BUMBLEBEE", # 'BUMBLEBEE': 70, 
        "GROUND BETTLE (HARPALUS SPP)", #'GROUND BETTLE (HARPALUS SPP)': 18, 
        "ANT", #'ANT': 10, 
        "PYRRHOCORIDAE", #'PYRRHOCORIDAE': 7, 
        "LONGICORN" #'LONGICORN': 5
    ]
    """
    rename_mapping_all_insecta = {
        'FROGHOPPER (CERCOPIDAE)': 'INSECTA',
        'SCARABAEIDAE': 'INSECTA',
        'GRAIN APHID (SITOBION AVENAE)':'GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID',
        'ROSE GRAIN APHID':'GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID',
        'FLY (MELANOSTOMA SPP.)':'FLY (DIPTERA)',
        'BIBIONID FLY (BIBIONIDAE)':'FLY (DIPTERA)',
        'MUSCIDAE (FLY)':'FLY (DIPTERA)',
        'LEAF MINERS':'FLY (DIPTERA)',
        'BEAN SEED FLY (DELIA SPP.)':'FLY (DIPTERA)',
        'FUNGUS GNAT (MYCETOPHILIDAE)':'FLY (DIPTERA)',
        'CHIRONOMID MIDGE (MALE)':'CHIRONOMID MIDGE'
    }   

    # rename_categories_for_coco(coco_folder, 'trainPEST.json', 'train_all_insect.json', rename_mapping_all_insecta)
    # rename_categories_for_coco(coco_folder, 'valPEST.json', 'val_all_insect.json', rename_mapping_all_insecta)

    """
    classes_name_list_for_only_pest = [
        "INSECTA (NOT CONCERNED)",
        "GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID",
        "POLLEN BEETLE (MELIGETHES SPP.)",
        "SNAIL",
        "CEREAL LEAF BEETLE (OULEMA MELANOPUS)",
        "CABBAGE STEM FLEA BETTLE",
        "SLUG",
        "CABBAGE WHITEFLY"
    ]
    """
    rename_mapping_pest_only = {
        'INSECTA':'INSECTA (NOT CONCERNED)',
        'FROGHOPPER (CERCOPIDAE)':'INSECTA (NOT CONCERNED)',
        'SCARABAEIDAE':'INSECTA (NOT CONCERNED)',
        'GRAIN APHID (SITOBION AVENAE)':'GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID',
        'ROSE GRAIN APHID':'GRAIN APHID (SITOBION AVENAE) or ROSE GRAIN APHID',
        'FLY (DIPTERA)':'INSECTA (NOT CONCERNED)',
        'FLY (MELANOSTOMA SPP.)':'INSECTA (NOT CONCERNED)',
        'BIBIONID FLY (BIBIONIDAE)':'INSECTA (NOT CONCERNED)',
        'MUSCIDAE (FLY)':'INSECTA (NOT CONCERNED)',
        'LEAF MINERS':'INSECTA (NOT CONCERNED)',
        'BEAN SEED FLY (DELIA SPP.)':'INSECTA (NOT CONCERNED)',
        'FUNGUS GNAT (MYCETOPHILIDAE)':'INSECTA (NOT CONCERNED)',
        'LADYBUG (COCCINELLIDAE)':'INSECTA (NOT CONCERNED)',
        'LADYBUG (COCCINELLIDAE) (PUPA)':'INSECTA (NOT CONCERNED)',
        'LADYBUG (COCCINELLIDAE) (LARVAE)':'INSECTA (NOT CONCERNED)',
        'SPIDER (ARANEUS SPP.)':'INSECTA (NOT CONCERNED)',
        'CHIRONOMID MIDGE (MALE)':'INSECTA (NOT CONCERNED)',
        'CHIRONOMID MIDGE':'INSECTA (NOT CONCERNED)',
        'BEETLE (COLEOPTERA)':'INSECTA (NOT CONCERNED)',
        'MOSQUITO':'INSECTA (NOT CONCERNED)',
        'WASP':'INSECTA (NOT CONCERNED)',
        'BEE':'INSECTA (NOT CONCERNED)',
        'HEMIPTERA (PLANT BUG)':'INSECTA (NOT CONCERNED)',
        'EARTHWORM':'INSECTA (NOT CONCERNED)',
        'BUMBLEBEE':'INSECTA (NOT CONCERNED)',
        'GROUND BETTLE (HARPALUS SPP)':'INSECTA (NOT CONCERNED)',
        'ANT':'INSECTA (NOT CONCERNED)',
        'PYRRHOCORIDAE':'INSECTA (NOT CONCERNED)',
        'LONGICORN':'INSECTA (NOT CONCERNED)',
    }   

    # rename_categories_for_coco(coco_folder, 'trainPEST.json', 'train_pest_only.json', rename_mapping_pest_only)
    # rename_categories_for_coco(coco_folder, 'valPEST.json', 'val_pest_only.json', rename_mapping_pest_only)

    # unified_image_format(coco_folder)
    # check_coco_folder("F:\\pest_data\\Multitask_or_multimodality\\coco_pest_2024_10_01")
    # check_image_file_in_VOC_and_COCO(voc_folder, "F:\\pest_data\\Multitask_or_multimodality\\coco_pest_2024_10_01")