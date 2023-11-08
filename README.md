# Repetitive Patterns on Textured 3D Surfaces Dataset

In this project we work with the dataset published in [A Benchmark Dataset for Repetitive Pattern Recognition on Textured 3D Surfaces](https://diglib.eg.org/handle/10.1111/cgf14352) focused in the task of detect and segment patterns. The original data is available in [the tugraz website](https://datasets.cgv.tugraz.at/pattern-benchmark/) where you can find more information about the paper and the structure of the dataset.

The dataset is composed by 82 differents 3D models of painted ancient Peruvial vessels, exhibiting different levels of repetitiveness in their surface patterns. The archeotypes exhibited in the surface were annotated by archaeologists using a specialized tool shared in the paper, obtaining a ground truth segmentation of the patterns. The format of the annotated data is a .json file with the following structure:

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

In this repository we provide a python script to convert the data in a COCO and YOLO standard format to be used in the training of deep learning models, and a python script to divide the data in train and validation with crops considering minimum visibility of the masks in the coordinates crops. 

The script to convert the data in COCO format and remove areas without segmentation is available in the file `convert.py`. You can run the following command for get the annotations:

```bash
python convert.py --crop_images
```

The script will create a folder `datasets` and a folder that contain the COCO format data with an `annotation.json` file and a `data` folder.

For get the annotations in the YOLO format you can run the following command:

```bash
python coco_to_yolo.py
```
We use the [JSON2YOLO format repository from ultralytics](https://github.com/ultralytics/JSON2YOLO.git) to convert the annotations from COCO to YOLO format with refactoring for the structure of datasets.


