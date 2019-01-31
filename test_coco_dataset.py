"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import shutil
import cv2
import numpy as np
from src.utils import *
import pickle
from src.yolo_net import Yolo

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--test_set", type=str, default="val")
    parser.add_argument("--year", type=str, default="2017", help="The year of dataset (2014 or 2017)")
    parser.add_argument("--root_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_coco")
    parser.add_argument("--output", type=str, default="predictions")

    args = parser.parse_args()
    return args


def test(opt):
    input_image_folder = os.path.join(opt.root_path, "images", "{}{}".format(opt.test_set, opt.year))
    anno_path = os.path.join(opt.root_path, "anno_pickle", "COCO_{}{}.pkl".format(opt.test_set, opt.year))
    id_list_path = pickle.load(open(anno_path, "rb"))
    id_list_path = list(id_list_path.values())
    output_folder = os.path.join(opt.output, "COCO_{}{}".format(opt.test_set, opt.year))
    colors = pickle.load(open("src/pallete", "rb"))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(80)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(80)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
    model.eval()

    for id in id_list_path:
        image_path = os.path.join(input_image_folder, id["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) == 0:
            continue
        else:
            predictions = predictions[0]
        output_image = cv2.imread(image_path)
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            color = colors[CLASSES.index(pred[5])]
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                output_image, pred[5] + ' : %.2f' % pred[4],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
            print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
        cv2.imwrite("{}/{}_prediction.jpg".format(output_folder, id["file_name"][:-4]), output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
