import os
import json
import argparse
import cv2
import numpy as np
import fiftyone as fo
from fiftyone.types.dataset_types import COCODetectionDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="datasets/coco_benchmark/")
args = parser.parse_args()

data_path = args.data_path

dataset = fo.Dataset.from_dir(
    data_path,
    dataset_type=COCODetectionDataset,
    name="benchmark",
    labels_path="annotations.json",
)

def recortar_bounding_box(ruta_imagen, x, y, ancho, alto, indice, nombre):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)

    # Calcular las coordenadas de la bounding box
    x1, y1 = x, y
    x2, y2 = x + ancho, y + alto

    # Recortar la bounding box
    bounding_box_recortada = imagen[y1:y2, x1:x2]

    # Guardar la bounding box recortada
    nombre = nombre.split(".")[0]
    # Hay que crear la carpeta recortes
    ruta_bounding_box_recortada = f"./recortes2/{nombre}_{indice}.png"
    cv2.imwrite(ruta_bounding_box_recortada, bounding_box_recortada)

    return ruta_bounding_box_recortada


for sample in dataset:
    image_name = sample.filepath.split("/")[-1]
    image_name_id = int(image_name.split("_")[0])

    # se transforma el sample a un diccionario
    dicc_info = sample.to_dict()
    # se obtiene las dimensiones de la imagen
    ancho_imagen, alto_imagen = dicc_info["metadata"]["width"],  dicc_info["metadata"]["height"]
    
    detecciones = bounding_box = dicc_info["detections"]["detections"]
    # para cada deteccion se recorta y se guarda la imagen
    for indice, deteccion in enumerate(detecciones):
        bounding_box = deteccion["bounding_box"]
        x, y = int(bounding_box[0] * ancho_imagen), int(bounding_box[1] * alto_imagen)
        ancho_bounding_box , alto_bounding_box = int(bounding_box[2] * ancho_imagen), int(bounding_box[3] * alto_imagen)

        ruta_imagen_original = f"./datasets/coco_benchmark/data/{image_name}"

        ruta_imagen_modificada = recortar_bounding_box(ruta_imagen_original, x, y, ancho_bounding_box, alto_bounding_box, indice, image_name)
        print(f"Imagen {image_name}_{indice} recortada")