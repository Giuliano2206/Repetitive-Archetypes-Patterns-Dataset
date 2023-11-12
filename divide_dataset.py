import argparse
import json
import os
import random

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from utils import polygon_to_bbox

parser = argparse.ArgumentParser(
    description=(
            'Divide the dataset in train and val, '
            'and apply transformations to the images'
    )
)

parser.add_argument(
    '--origin_coco_path',
    type=str,
    default='./coco_benchmark',
    help='Path to the original COCO benchmark dataset'
)

parser.add_argument(
    '--path_crops_benchmark',
    type=str,
    default='./coco_benchmark_divided',
    help='Path to save the COCO benchmark dataset'
)

parser.add_argument(
    '--val_size',
    type=float,
    default=0.2,
    help='Percentage of the images to use as val'
)

parser.add_argument(
    '--cat_unique',
    type=bool,
    default=False,
    help='Use only one category'
)

parser.add_argument(
    '--output_original',
    type=bool,
    default=False,
    nargs='?',
    const=True,
    help='Output original images in the val set'
)

parser.add_argument(
    '--min_visibility',
    type=float,
    default=0.1,
    help='Minimum visibility to consider an annotation'
)

parser.add_argument(
    '--output_zero_shot',
    type=bool,
    default=False,
    nargs='?',
    const=True,
    help='Output images with zero-shot annotations'
)

def create_dataset_crops(coco: COCO, path_origin_benchmark, path_crops_benchmark):
    images = coco.loadImgs(coco.getImgIds())
    total_images = len(images)
    images, ann = crop_images_dataset(coco, 
                                      images, 
                                      path_origin_benchmark, 
                                      path_crops_benchmark, 
                                      total_images=total_images)
    coco.dataset["images"] = images
    coco.dataset['annotations'] = ann
    coco_path = os.path.join(path_crops_benchmark, 'annotations.json')
    d = json.dumps(coco.dataset)
    with open(coco_path, 'w') as f:
        f.write(d)

def binary_mask_to_rle(mask):
    mask_poly = mask_util.encode(np.asfortranarray(mask))
    mask_poly['counts'] = mask_poly['counts'].decode('utf-8')
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.reshape(-1).tolist() for c in contours]
    return mask_poly, contours

def get_annotation_from_transform(annotation, bin_mask, image_id, to_rle=False, min_visibility=0.1):
    new_an = annotation.copy()
    area_org = new_an['area']
    if to_rle:
        segmentation = mask_util.encode(np.asfortranarray(bin_mask))
        segmentation['counts'] = segmentation['counts'].decode('utf-8')
        bbox = mask_util.toBbox(segmentation)
        bbox = [float(round(x,1)) for x in bbox]
    else:
        segmentation, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = np.reshape(segmentation[0], (-1,2))
        bbox = polygon_to_bbox(segmentation)
    new_area = bbox[2] * bbox[3]
    if area_org > 0 and new_area / area_org > min_visibility:
        new_an['bbox'] = bbox
        new_an['area'] = int(new_area)
        new_an['segmentation'] = [segmentation.reshape(-1).tolist()]
        new_an['image_id'] = image_id
        return new_an
    else:
        return None

def get_area(segmentation, h, w) -> float:
    # get the area of the segmentation
    rle = mask_util.frPyObjects(segmentation, h, w)
    area = mask_util.area(rle)
    return int(area[0])

def crop_images_dataset(coco, images, path_origin_benchmark, path_crops_benchmark, total_images):
    new_anns = []
    new_images = []
    id_ann = 0
    for img in tqdm(images, desc='Images Cropped', leave=False, position=0, ncols=100, unit='img'):
        file_name = img['file_name']
        h_org, w_org = img['height'], img['width']
        img_path = os.path.join(path_origin_benchmark, file_name)
        paths, sizes = crop_image(img_path, cropped_image_dir=path_crops_benchmark)
        img_1 = dict(
            file_name=os.path.basename(paths[0]),
            height=sizes[0][0],
            width=sizes[0][1],
            id=img['id'],
        )
        img_2 = dict(
            file_name=os.path.basename(paths[1]),
            height=sizes[1][0],
            width=sizes[1][1],
            id=img['id'] + total_images,
        )
        new_images.append(img_1)
        new_images.append(img_2)
        anns_img = coco.getAnnIds(imgIds=img['id'])
        anns_img = coco.loadAnns(anns_img)
        for an in anns_img:
            image_id = an['image_id']
            segmentation = an['segmentation']
            mask_bin = np.zeros((h_org, w_org), dtype=np.uint8)
            segmentation = np.array(segmentation).reshape(-1,2).astype(np.int32)
            mask_bin = cv2.fillPoly(mask_bin, [segmentation], 1)
            mask_bin = mask_bin.astype(bool)
            left_side = mask_bin[:, :w_org//2]
            right_side = mask_bin[:, w_org//2:]
            if left_side.any():
                new_ann = get_annotation_from_transform(
                    annotation=an, 
                    bin_mask=left_side,
                    image_id=an['image_id']    
                )
                if new_ann is not None:
                    new_anns.append(new_ann)
            if right_side.any():
                new_image_id = image_id + total_images
                new_ann = get_annotation_from_transform(
                    annotation=an,
                    bin_mask=right_side,
                    image_id=new_image_id
                )
                if new_ann is not None:
                    new_anns.append(new_ann)
    id_ann = 0
    for ann in new_anns:
        ann['id'] = id_ann
        id_ann += 1
    return new_images, new_anns
    

def crop_image(img_path, cropped_image_dir, parts=2):
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    c1 = image[:, :w//parts, :]
    c2 = image[:, w//parts:, :]
    basename = os.path.basename(img_path)
    n1 = f'left_{basename}'
    n2 = f'right_{basename}'
    path_1 = os.path.join(cropped_image_dir, "data", n1)
    path_2 = os.path.join(cropped_image_dir, "data", n2)
    cv2.imwrite(path_1, c1)
    cv2.imwrite(path_2, c2)
    return [path_1, path_2], [c1.shape[:2], c2.shape[:2]]


def get_unique_images(images, val_size, seed=42):
    random.seed(seed)
    images_filename = [img['file_name'].split('_', 1)[1] for img in images]
    images_filename = list(set(images_filename))
    total_images = len(images_filename)
    images_filename.sort()
    val_images_sample = random.sample(images_filename, int(total_images * val_size))
    val_images_sample.sort()
    return val_images_sample

def get_val_mid_sample(images, val_size, seed=42):
    val_images_sample = get_unique_images(images, val_size, seed)
    left_right = ['left', 'right']
    # odd images are left images, even images are right images
    for i in range(len(val_images_sample)):
        val_images_sample[i] = left_right[i % 2] + '_' + val_images_sample[i]
    return val_images_sample

def get_val_zero_sample(images, val_size, seed=42):
    val_images_sample = get_unique_images(images, val_size, seed)
    left_right = ['left', 'right']
    val_left_right = []
    for prefix in left_right:
        for img in val_images_sample:
            print(prefix + '_' + img)
            val_left_right.append(prefix + '_' + img)
    return val_left_right

def copy_images_to_folders(path_crops_benchmark, images, folder):
    os.makedirs(os.path.join(path_crops_benchmark, folder, "data"), exist_ok=True)
    for img in images:
        file_name = img['file_name']
        src = os.path.join(path_crops_benchmark, "data", file_name)
        dst = os.path.join(path_crops_benchmark, folder, "data", file_name)
        os.system(f'cp {src} {dst}')

def create_annotation_file(ann_data, path_crops_benchmark, folder, annotation_file):
    path_ann = os.path.join(path_crops_benchmark, folder, annotation_file)
    with open(path_ann, 'w') as f:
        f.write(json.dumps(ann_data))

def divide_train_val(path_crops_benchmark, 
                     val_size=0.1,
                     annotation_file='annotations.json', 
                     cat_unique=False,
                     output_original=False,
                     output_zero_shot=False, 
                     ):
    random.seed(42)
    train_folder = 'train'
    val_folder = 'val'
    coco_path = os.path.join(path_crops_benchmark, annotation_file)
    with open(coco_path) as f:
        coco = json.load(f)
    images = coco["images"]
    annotations = coco['annotations']
    if cat_unique:
        for ann in annotations:
            ann['category_id'] = 1
    images_filename = [img['file_name'] for img in images]
    images_filename.sort()
    if output_original:
        val_images_sample = random.sample(images_filename, int(len(images_filename) * val_size))
    elif output_zero_shot:
        val_images_sample = get_val_zero_sample(images, val_size)
    else:
        val_images_sample = get_val_mid_sample(images, val_size)
    val_images = [img for img in images if img['file_name'] in val_images_sample]
    train_images = [img for img in images if img['file_name'] not in val_images_sample]
    # get the annotations for train and val
    val_images_id = [img['id'] for img in val_images]
    train_images_id = [img['id'] for img in train_images]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_images_id]
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_images_id]

    if cat_unique:
        categories = [{'id': 1, 'name': 'pattern'}]
    else:
        categories = coco['categories']

    # create the coco dict for train and val
    train_coco = dict(
        images=train_images,
        annotations=train_annotations,
        categories=categories
    )
    val_coco = dict(
        images=val_images,
        annotations=val_annotations,
        categories=categories
    )
    copy_images_to_folders(path_crops_benchmark, 
                           train_images, 
                           train_folder)
    
    copy_images_to_folders(path_crops_benchmark,
                            val_images,
                            val_folder)

    # create the annotations files
    create_annotation_file(train_coco,
                            path_crops_benchmark,
                            train_folder,
                            annotation_file
                           )
    create_annotation_file(val_coco,
                            path_crops_benchmark,
                            val_folder,
                            annotation_file
                           )


def default_transform():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    ], 
    )
    return transform

def albumentations_masks(image, masks, classes_labels, transform):
    transformed = transform(image=image, masks=masks, class_labels=classes_labels) 
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']
    transformed_labels = transformed['class_labels']
    return transformed_image, transformed_masks, transformed_labels

def apply_albumentations(coco_dataset, images, annotations, cats, path_crops_benchmark, folder, new_folder, transform=default_transform()):
    total_images = len(images)
    new_image_id = total_images + 1
    new_ann_id = len(annotations) + 1
    images_transformed = []
    anns_transformed = []
    for i, image in enumerate(tqdm(images, desc='Images', leave=False, position=0, ncols=100, unit='img')):
        print(f'Processing image {i}')
        file_name = image['file_name']
        img_path = os.path.join(path_crops_benchmark, folder, "data", file_name)
        image_bgr = cv2.imread(img_path)
        name_transformation = 't1'
        file_name_transformation = f'{name_transformation}_{file_name}'
        img_path_transformation = os.path.join(path_crops_benchmark, new_folder, "data", f'{file_name_transformation}')
        h, w = image_bgr.shape[:2]
        ann = [an for an in annotations if an['image_id'] == image['id']]
        mask_poly = [an['segmentation'] for an in ann]
        class_labels = [an['category_id'] for an in ann]
        mask_binary = [coco_dataset.annToMask(an) for an in ann]
        #bboxes = [an['bbox'] for an in ann]
        #transform_image, transform_bboxes, transformed_labels = albumentations_masks(image_bgr, bboxes, class_labels, transform)
        transform_image, transform_masks, transformed_labels = albumentations_masks(image_bgr, mask_binary, class_labels, transform)
        h_new, w_new = transform_image.shape[:2]
        new_anns = []
        for j, mask in enumerate(transform_masks):
            mask_poly = mask_util.encode(np.asfortranarray(mask))
            mask_poly['counts'] = mask_poly['counts'].decode('utf-8')
            polys = mask_util.decode(mask_poly)
            polys = polys.astype(np.float32)
            polys = polys.reshape(-1).tolist()
            bbox = mask_util.toBbox(mask_poly)
            bbox = [float(round(x,1)) for x in bbox]
            area = mask_util.area(mask_poly)
            area = int(area)
            new_anns.append(dict(
                area=area,
                iscrowd=0,
                image_id=new_image_id,
                id=new_ann_id,
                category_id=transformed_labels[j],
                segmentation=polys,
                bbox=bbox,
            ))
            new_ann_id += 1
        images_transformed.append(dict(
            file_name=file_name_transformation,
            height=h_new,
            width=w_new,
            id=new_image_id,
        ))
        cv2.imwrite(img_path_transformation, transform_image)
        new_image_id += 1
        anns_transformed.extend(new_anns)
    images_new = images_transformed
    annotations_new = anns_transformed
    coco_path = os.path.join(path_crops_benchmark, new_folder, 'annotations.json')
    d = dict(
        images=images_new,
        annotations=annotations_new,
        categories=cats,
    )
    with open(coco_path, 'w') as f:
        f.write(json.dumps(d))


def increase_dataset(path_crops_benchmark,
                     train_folder='train',
                     val_folder='val',
                     annotation_file='annotations.json',
                     cat_unique=True,
                     ):
    coco_train = COCO(os.path.join(path_crops_benchmark, train_folder, annotation_file))
    coco_val = COCO(os.path.join(path_crops_benchmark, val_folder, annotation_file))
    images_train = coco_train.dataset["images"]
    images_val = coco_val.dataset["images"]
    annotations_train = coco_train.dataset['annotations']
    annotations_val = coco_val.dataset['annotations']
    
    cats_train = coco_train.dataset['categories']
    cats_val = coco_val.dataset['categories']

    train_folder_new = f'{train_folder}_augmented'
    val_folder_new = f'{val_folder}_augmented'

    os.makedirs(os.path.join(path_crops_benchmark, train_folder_new, "data"), exist_ok=True)
    os.makedirs(os.path.join(path_crops_benchmark, val_folder_new, "data"), exist_ok=True)

    apply_albumentations(coco_train, images_train, annotations_train, cats_train, path_crops_benchmark, train_folder, train_folder_new)
    apply_albumentations(coco_val, images_val, annotations_val, cats_val, path_crops_benchmark, val_folder, val_folder_new)

    # copy images to new data folder
    os.system(f'cp {os.path.join(path_crops_benchmark, train_folder, "data", "*")} {os.path.join(path_crops_benchmark, train_folder_new, "data")}')
    os.system(f'cp {os.path.join(path_crops_benchmark, val_folder, "data", "*")} {os.path.join(path_crops_benchmark, val_folder_new, "data")}')


def main(origin_coco_path, 
         path_crops_benchmark, 
         val_size, 
         cat_unique=False, 
         output_original=False,
         output_zero_shot=False,
         crop_in_parts=True,
        ):
    os.makedirs(os.path.join(path_crops_benchmark, "data"), exist_ok=True)
    origin_annotations = os.path.join(origin_coco_path, 'annotations.json')
    origin_images = os.path.join(origin_coco_path, "data")
    coco = COCO(origin_annotations)
    if crop_in_parts:
        create_dataset_crops(coco, origin_images, path_crops_benchmark)
    divide_train_val(path_crops_benchmark,
                        val_size=val_size,
                        output_original=output_original,
                        cat_unique=cat_unique,
                        output_zero_shot=output_zero_shot
                    )

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_folder = './datasets'
    origin_coco_path = os.path.join(dataset_folder, 'coco_benchmark')
    path_crops_benchmark = os.path.join(dataset_folder, 'coco_benchmark_divided')
    main(origin_coco_path, 
        path_crops_benchmark,
        val_size=0.1, 
        cat_unique=args.cat_unique, 
        output_original=args.output_original,
        output_zero_shot=args.output_zero_shot
        )

