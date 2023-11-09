import glob
import shutil
from pathlib import Path

import numpy as np
from PIL import ExifTags

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def polygon_to_bbox(polygon):
    x1, y1 = np.min(polygon, axis=0)
    x2, y2 = np.max(polygon, axis=0)
    coords = [x1, y1, x2 - x1, y2 - y1]
    coords = [float(round(x,1)) for x in coords]
    return coords


def split_files(out_path, file_name, prefix_path=''):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(f'{out_path}_{key}.txt', 'a') as file:
                for i in item:
                    file.write('%s%s\n' % (prefix_path, file_name[i]))


def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices


def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def write_data_data(fname='data.data', nc=80):
    # write darknet *.data file
    lines = ['classes = %g\n' % nc,
             'train =../out/data_train.txt\n',
             'valid =../out/data_test.txt\n',
             'names =../out/data.names\n',
             'backup = backup/\n',
             'eval = coco\n']

    with open(fname, 'a') as f:
        f.writelines(lines)


def image_folder2file(folder='images/'):  # from utils import *; image_folder2file()
    # write a txt file listing all imaged in folder
    s = glob.glob(f'{folder}*.*')
    with open(f'{folder[:-1]}.txt', 'w') as file:
        for l in s:
            file.write(l + '\n')  # write image list

def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x
