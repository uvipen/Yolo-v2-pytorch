"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import json
import pickle
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser("Parsing MS COCO dataset")
    parser.add_argument("--input", type=str, default="data/COCO")
    parser.add_argument("--type", type=str, default="val2014")
    parser.add_argument("--output", type=str, default="data/COCO/anno_pickle")
    args = parser.parse_args()
    return args

def main(opt):
    ann_file = '{}/annotations/instances_{}.json'.format(opt.input, opt.type)
    dataset = json.load(open(ann_file, 'r'))
    image_dict = {}
    invalid_anno = 0

    for image in dataset["images"]:
        if image["id"] not in image_dict.keys():
            image_dict[image["id"]] = {"file_name": image["file_name"], "objects": []}

    for ann in dataset["annotations"]:
        if ann["image_id"] not in image_dict.keys():
            invalid_anno += 1
            continue
        image_dict[ann["image_id"]]["objects"].append(
            [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][0] + ann["bbox"][2]),
             int(ann["bbox"][1] + ann["bbox"][3]), ann["category_id"]])

    pickle.dump(image_dict, open(opt.output + os.sep + 'COCO_{}.pkl'.format(opt.type), 'wb'))
    print ("There are {} invalid annotation(s)".format(invalid_anno))


if __name__ == "__main__":
    opt = get_args()
    main(opt)
