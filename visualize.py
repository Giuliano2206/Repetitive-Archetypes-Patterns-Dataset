import os
import json

import fiftyone as fo
from fiftyone.types.dataset_types import COCODetectionDataset

data_path = "datasets/coco_benchmark/"

dataset = fo.Dataset.from_dir(
    data_path,
    dataset_type=COCODetectionDataset,
    name="benchmark",
    labels_path="annotations.json",
)
culture_file = "datasets/datasetCulture/culture.json"
shape_file = "datasets/datasetCulture/shape.json"

if os.path.exists(culture_file):
    with open("datasets/datasetCulture/culture.json") as f:
        culture = json.load(f)

if os.path.exists(shape_file):
    with open("datasets/datasetCulture/shape.json") as f:
        shape = json.load(f)

for sample in dataset:
    image_name = sample.filepath.split("/")[-1]
    image_name_id = int(image_name.split("_")[0])
    for culture_name, culture_id in culture.items():
        if image_name_id in culture_id:
            sample['culture'] = culture_name
    for shape_name, shape_id in shape.items():
        if image_name_id in shape_id:
            sample['shape'] = shape_name
    sample.save()

dataset.save()

session = fo.launch_app(dataset)
session.wait()

exit()
