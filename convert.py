import argparse
import glob
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm

from utils import *

parser = argparse.ArgumentParser(
    description=(
            'Convert pattern-benchmark-v0.1 to COCO format, '
            'crop the images and save the annotations.json file'
    )
)
parser.add_argument(
    '--path_dataset', 
    type=str, 
    default='./pattern-benchmark-v0.1',
    help='Path to the RPT3DS dataset'
)
parser.add_argument(
    '--path_coco_benchmark', 
    type=str, 
    default='./coco_benchmark',
    help='Path to save the COCO benchmark dataset'
)
parser.add_argument(
    '--original_images', 
    type=bool, 
    default=False,
    nargs='?',
    const=True,
    help=(
        'Creates COCO benchmark dataset with the images cropped '
        'and the transformed annotations.json file in '
        'the path_coco_benchmark folder'
    )
)
parser.add_argument(
    '--to_rle',
    type=bool,
    default=False,
    nargs='?',
    const=True,
    help='Converts the segmentation mask to RLE format'
)
parser.add_argument(
    '--skip_test',
    type=bool,
    default=False,
    nargs='?',
    const=True,
    help='Skip the test images'
)


def crop_poly_traslation(poly, left_crop):
    crop_poly = []
    for i in range(len(poly)):
        x, y = poly[i]
        x = x - left_crop
        x = max(0, x)
        crop_poly.append([x, y])
    return crop_poly

def get_polygon_points( data: dict, 
                        img_path: str, 
                        image_id: int,
                        image_shape: tuple,
                        left_coord = 1000,
                        last_category_id = 0,
                        last_annotation_id = 0,
                        to_rle = False
                       ):
    # get the polygon points
    annotations = data['annotations']
    cat_id = last_category_id
    ann_id = last_annotation_id
    basename = os.path.basename(img_path)
    categories_patterns = []
    anns = []
    for pattern in annotations:
        pattern_id = pattern['patternId']
        selections = pattern['selections']
        for shape in selections:
            segm_info = {}
            polygon = shape['polygonPts']
            polygon_translated = crop_poly_traslation(polygon, left_coord)
            bbox = polygon_to_bbox(polygon_translated)
            segmentation = [np.array(polygon_translated).reshape(-1).tolist()]
            if to_rle:
                rle = mask_util.frPyObjects(segmentation, image_shape[0], image_shape[1])
                rle = mask_util.merge(rle)
                rle['counts'] = rle['counts'].decode('utf-8')
                segmentation = rle
            segm_info = dict(
                id=ann_id,
                image_id=image_id,
                category_id=cat_id,
                segmentation=segmentation,
                bbox=bbox,
                #area=mask_util.area(mask_util.frPyObjects([polygon_pts], h_img, w_img)),
                area=bbox[2] * bbox[3],
                iscrowd=0,
                bbox_mode=1,
            )
            anns.append(segm_info)
            ann_id += 1
        categories_patterns.append(
            dict(
                id=cat_id, 
                name=f"{basename.split('.')[0]}_{pattern_id}"
            ))
        cat_id += 1
    return anns, categories_patterns, cat_id, ann_id

def get_coords_crops(image_data: dict, margin=20, width_data=5000):
    """
    Crops the image to remove not annotated areas
    Returns: coordenates to crop the image, in the format (left, right)
    """
    annotations = image_data['annotations']
    all_polygon_points = []
    for pattern in annotations:
        for shape in pattern['selections']:
            all_polygon_points.extend(shape['polygonPts'])
    all_polygon_points = np.array(all_polygon_points).reshape(-1, 2)
    x1_min, _ = np.min(all_polygon_points, axis=0)
    x2_max, _ = np.max(all_polygon_points, axis=0)
    left_crop = x1_min - margin
    right_crop = width_data - x2_max - margin
    return [int(left_crop), int(right_crop)]


def crop_image(img_path: str, 
               cropped_image_dir: str, 
               cropped_image_dir_test: str, 
               crop_coords: list[int, int],
                skip_test: bool = False
                ):
    image = cv2.imread(img_path)
    basename = os.path.basename(img_path)
    cropped_image_path = os.path.join(cropped_image_dir, basename)
    width = image.shape[1]
    cropped_image = image[:, crop_coords[0]:width - crop_coords[1], :]
    left_side = image[:, :crop_coords[0], :]
    right_side = image[:, width - crop_coords[1]:, :]
    cv2.imwrite(cropped_image_path, cropped_image)
    if not skip_test:
        cv2.imwrite(f'{cropped_image_dir_test}/left_{basename}', left_side)
        cv2.imwrite(f'{cropped_image_dir_test}/right_{basename}', right_side)
    w_cropped, h_cropped = cropped_image.shape[1], cropped_image.shape[0]
    return cropped_image_path, (w_cropped, h_cropped)

def build_coco_json(folders,
                    path_data, 
                    path_test, 
                    original_images,
                    to_rle=False,
                    skip_test=False
                    ):
    images_json = []
    categories = []
    annotations = []
    category_id = 1
    annotation_id = 0
    image_id = 0
    for i, folder in enumerate(tqdm(folders, 
                                    desc='Images', 
                                    leave=False, 
                                    position=0, 
                                    ncols=100, 
                                    unit='img')):
        img_path = glob.glob(folder + '/*_flat.png')[0]
        json_file = glob.glob(folder + '/*.json')[0]
        with open(json_file) as file:
            segmentation_data = json.load(file)
            crop_coords = get_coords_crops(segmentation_data)
            basename = os.path.basename(img_path)
            if not original_images:
                crop_path, (w,h) = crop_image(img_path, 
                                              cropped_image_dir=path_data, 
                                              cropped_image_dir_test=path_test,
                                              crop_coords=crop_coords,
                                              skip_test=skip_test
                )
            else:
                crop_path = img_path
                image = cv2.imread(crop_path)
                cv2.imwrite(os.path.join(path_data, basename), image)
                w, h = image.shape[1], image.shape[0]
            image_info = dict(
                file_name=basename,
                height=h,
                width=w,
                id=image_id,
            )
            basename = os.path.basename(json_file)
            anns, cat_pat, new_cat, new_ann = get_polygon_points(
                        data=segmentation_data, 
                        img_path=crop_path,
                        image_shape=(h, w),
                        image_id=image_id,
                        last_category_id=category_id,
                        last_annotation_id=annotation_id,
                        left_coord=crop_coords[0],
                        to_rle=to_rle
            )
            category_id = new_cat
            annotation_id = new_ann
            images_json.append(image_info)
            categories.extend(cat_pat)
            annotations.extend(anns)
            image_id += 1
    return images_json, annotations, categories

def main(path_dataset, path_coco_benchmark, original_images, to_rle=False, skip_test=False):
    path_test = f'{path_coco_benchmark}_test'
    path_data = os.path.join(path_coco_benchmark, "data")
    os.makedirs(path_data, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)
    folders = glob.glob(path_dataset + '/*')
    exclude_images = ['0056']
    folders = [folder for folder in folders if os.path.basename(folder) not in exclude_images]
    images_json, annotations, categories = build_coco_json(
        folders, path_data, path_test, original_images, to_rle, skip_test
    )
    info = dict(
        year=2021,
        version=1,
        description=("Images 2D of repetitive patterns on textured 3D surfaces captured " 
                     "with a structured-light scanner by the Josefina Ramos de Cox museum in Lima, Perú"
                    ),
        contributor="Josefina Ramos de Cox museum",
        url="https://datasets.cgv.tugraz.at/pattern-benchmark/",
        date_created="2021/06/20",
    )
    licenses = [
        dict(
            id=1,
            name="Attribution-NonCommercial-ShareAlike 4.0 International",
            url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        )
    ]
    coco_data = dict(
        info=info,
        licenses=licenses,
        images=images_json,
        categories=categories,
        annotations=annotations,
    )
    with open(os.path.join(path_coco_benchmark, 'annotations.json'), 'w') as file:
        json.dump(coco_data, file)

if __name__ == '__main__':
    args = parser.parse_args()
    path_dataset = os.path.join('./datasets/', args.path_dataset)
    path_coco_benchmark = os.path.join('./datasets/', args.path_coco_benchmark)
    main(path_dataset, path_coco_benchmark, args.original_images, args.to_rle, args.skip_test)