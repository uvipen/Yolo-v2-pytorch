"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from src.coco_dataset import COCODataset
from src.utils import *
from src.loss import YoloLoss
from src.yolo_net import Yolo
from tensorboardX import SummaryWriter
import shutil


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=10, help="The number of images per batch")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=160)
    parser.add_argument("--test_interval", type=int, default=5, help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_coco")
    parser.add_argument("--log_path", type=str, default="tensorboard/yolo_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    learning_rate_schedule = {"0": 1e-5, "5": 1e-4,
                              "80": 1e-5, "110": 1e-6}
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    training_set = []
    training_generator = []
    training_set.append(COCODataset(opt.data_path, "2014", "train", opt.image_size))
    training_set.append(COCODataset(opt.data_path, "2014", "val", opt.image_size))
    training_set.append(COCODataset(opt.data_path, "2017", "train", opt.image_size))
    training_generator.append(DataLoader(training_set[0], **training_params))
    training_generator.append(DataLoader(training_set[1], **training_params))
    training_generator.append(DataLoader(training_set[2], **training_params))

    test_set = COCODataset(opt.data_path, "2017", "val", opt.image_size, is_training=False)
    test_generator = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(training_set[0].num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(training_set[0].num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
    # The following line will re-initialize weight for the last layer, which is useful
    # when you want to retrain the model based on my trained weights. if you uncomment it,
    # you will see the loss is already very small at the beginning.
    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)
    log_path = os.path.join(opt.log_path, "{}".format("2014and2017"))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    if torch.cuda.is_available():
        writer.add_graph(model.cpu(), torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
        model.cuda()
    else:
        writer.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
    criterion = YoloLoss(training_set[0].num_classes, model.anchors, opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = 0
    for generator in training_generator:
        num_iter_per_epoch += len(generator)
    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for generator in training_generator:
            for iter, batch in enumerate(generator):
                image, label = batch
                if torch.cuda.is_available():
                    image = Variable(image.cuda(), requires_grad=True)
                else:
                    image = Variable(image, requires_grad=True)
                optimizer.zero_grad()
                logits = model(image)
                loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
                loss.backward()
                optimizer.step()
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    loss_coord,
                    loss_conf,
                    loss_cls))
                writer.add_scalar('Train/Total_loss', loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Coordination_loss', loss_coord, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Confidence_loss', loss_conf, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Class_loss', loss_cls, epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_coord_ls = []
            loss_conf_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_image, te_label = te_batch
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_image = te_image.cuda()
                with torch.no_grad():
                    te_logits = model(te_image)
                    batch_loss, batch_loss_coord, batch_loss_conf, batch_loss_cls = criterion(te_logits, te_label)
                loss_ls.append(batch_loss * num_sample)
                loss_coord_ls.append(batch_loss_coord * num_sample)
                loss_conf_ls.append(batch_loss_conf * num_sample)
                loss_cls_ls.append(batch_loss_cls * num_sample)
            te_loss = sum(loss_ls) / test_set.__len__()
            te_coord_loss = sum(loss_coord_ls) / test_set.__len__()
            te_conf_loss = sum(loss_conf_ls) / test_set.__len__()
            te_cls_loss = sum(loss_cls_ls) / test_set.__len__()
            print("Epoch: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss,
                te_coord_loss,
                te_conf_loss,
                te_cls_loss))
            writer.add_scalar('Test/Total_loss', te_loss, epoch)
            writer.add_scalar('Test/Coordination_loss', te_coord_loss, epoch)
            writer.add_scalar('Test/Confidence_loss', te_conf_loss, epoch)
            writer.add_scalar('Test/Class_loss', te_cls_loss, epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                # torch.save(model, opt.saved_path + os.sep + "trained_yolo_coco")
                torch.save(model.state_dict(), opt.saved_path + os.sep + "only_params_trained_yolo_coco")
                torch.save(model, opt.saved_path + os.sep + "whole_model_trained_yolo_coco")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    # writer.export_scalars_to_json(log_path + os.sep + "all_logs.json")
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
