# Repetitive Patterns on Textured 3D Surfaces Dataset

In this project, we work with the dataset published in [A Benchmark Dataset for Repetitive Pattern Recognition on Textured 3D Surfaces](https://diglib.eg.org/handle/10.1111/cgf14352) focused on the task of detecting and segmenting patterns. The original data is available on [the TUGRAZ website](https://datasets.cgv.tugraz.at/pattern-benchmark/) where you can find more information about the paper and the structure of the dataset. Also, you can get complementary information about the cultures and the archetypes of the patterns in the [SHREC 2021: Retrieval of Cultural Heritage Objects repository](https://github.com/ivansipiran/shrec2021-cultural-heritage).

The dataset is composed of 82 different 3D models of painted ancient Peruvian vessels and 82 images 2D of textures, exhibiting different levels of repetitiveness in their surface patterns. The archetypes exhibited on the surface were annotated by archaeologists using a specialized tool shared in the paper, obtaining a ground truth segmentation of the patterns. The format of the annotated data is a JSON file with the following structure:

```json
{
    "annotations": [
        {
            "fileName": "<id_image>.pat<pattern_id>.png",
            "foldSymmetry": 0,
            "patternId": <pattern_id>,
            "selections": [
              {
                  "faceIdxs": [int...],
                  "flipped": bool,
                  "polygonPts": [[int, int] ...],
                  "rotation": float,
                  "scale": float,
              }
          ],
       }
   ]
}
```

In this repository, we transform the 2D image textures and provide the `convert.py` and `coco_to_yolo.py` scripts to convert the data in a COCO and YOLO standard format respectively to be used in the training of deep learning models. Also, with the `divide_dataset.py` script, you can divide the data in train and validation with crops considering the minimum area of the patterns.

## Download

The data can be downloaded directly using the `download_data.sh` script. 

```bash
bash download_data.sh
```
Also, you can download the minimized original data, the COCO and the YOLO data in the release section of this repository.

## Train and Validation data

The transformation process is composed of three steps. First, we crop the images to remove the areas without segmentation. We only consider the image width to not remove relevant regions of the original image. This step saves images that do not contain segmentation in the `datasets/coco_benchmark_test/` folder.

Second, we convert the data to COCO format. This step is necessary to get the annotations in the YOLO format. The COCO format data is saved in the `datasets/coco_benchmark/` folder.

Finally, we can divide the data in train and validation duplicating the data by cropping the image in 2 parts in the center, considering the minimum bounding box area visibility of the patterns. After this step, the images can be divided into a set of `train` and `val`. The division process follows the next rules:

* The images are divided in the center ignoring the masks with less than 10% of the area original of the bounding box pattern. We do not want patterns that are too small. 
* The original images (before the crop) are divided into train and validation with a proportion of 60% and 40% respectively. Once are divided the images, the mid-images of the validation set are moved to the train set to avoid the duplication of the images in validation. The strategy is graphically represented in the following image:

![Division process](./images/unique_images_strategy.png)

* In another strategy, for experiments `zero-shot`, the train and validation sets are built with the initial proportion of 60% and 40% respectively. The difference is that the images are duplicated, but with a small size, and the images of the validation set are not moved to the train set. The strategy is graphically represented in the following image:

![Division zero shot](./images/zero_shot_strategy.png)

## Convert to COCO and YOLO format

The script to convert the data in COCO format and remove areas without segmentation is available in the file `convert.py`. You can run the following command to get the annotations:

```bash
python convert.py
```

The script will create the folder that contains the COCO format data with an `annotation.json` file and a `data` folder in the `datasets` folder. If you want to convert the dataset considering the regions without annotation you can add the parameter `--original_images`.

To split the data in train and validation, you can run the following command:

```bash
python divide_dataset.py
```
Once you obtain the COCO formatted data, you can get the annotations in the YOLO format by running the following command:

```bash
python coco_to_yolo.py
```
**For transforming COCO to YOLO format, is necessary that the segmentation in the annotation file is in polygon points format. RLE format is not supported.** 

We use the [JSON2YOLO format repository from ultralytics](https://github.com/ultralytics/JSON2YOLO.git) to convert the annotations from COCO to YOLO format with refactoring for the structure of datasets.

## Citation

* [Repetitive Patterns on Textured 3D Surfaces](https://datasets.cgv.tugraz.at/pattern-benchmark/) [![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/) 2021 by Lengauer, Stefan and Sipiran, Ivan and Preiner, Reinhold and Schreck, Tobias and Bustos, Benjamin is licensed under [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
* [![DOI](https://zenodo.org/badge/186122711.svg)](https://zenodo.org/badge/latestdoi/186122711)