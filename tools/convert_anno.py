import json
from math import pi
import os

def box_fromator(box):
    box[0] = box[0] / pi * 180 + 180
    box[1] = 90 - box[1] / pi * 180

def convert_format(anno_path):
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    for single_anno in anno['annotations']:
        box_fromator(single_anno['bbox'])
    with open(anno_path, 'w') as f:
        json.dump(anno, f)

if __name__ == '__main__':
    anno_dir = 'datasets/PANDORA/annotations'
    anno_names = ['instances_train2017.json', 'instances_val2017.json']

    for anno_name in anno_names:
        anno_path = os.path.join(anno_dir, anno_name)
        convert_format(anno_path)