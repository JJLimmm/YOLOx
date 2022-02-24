#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# Copyright (c) Bharath Hariharan.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        # not parsing pose since redundant.
        try:
            obj_struct["truncated"] = int(obj.find("truncated").text)
        except:
            obj_struct["truncated"] = 0  # default
        try:
            obj_struct["difficult"] = int(obj.find("difficult").text)
        except:
            obj_struct["difficult"] = 0  # default
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
        # save
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    splitlines = [
        [" ".join(x[:-5])] + x[-5:] for x in splitlines
    ]  # -5 index protects against unexpected spaces in image_id.
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def det_image_visualization(detpath, imagepath, classname):

    """ Logging helper function to visualize detections made during evaluation.

    Returns:
        figure: matplotlib figure containing grid of annotated images.
    """

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) < 8:
        # not enough detections to create a meaningful plot. Return a blank figure instead of None
        figure = plt.figure(figsize=(20, 10))
        return figure

    splitlines = [x.strip().split(" ") for x in lines]
    splitlines = [
        [" ".join(x[:-5])] + x[-5:] for x in splitlines
    ]  # -5 index protects against unexpected spaces in image_id.
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = BB.astype(np.int32)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidence = confidence[sorted_ind]
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # collect all instances of boxes with matching img ids:
    image_id_to_indexes = {}
    for index in range(len(image_ids)):
        if image_ids[index] not in image_id_to_indexes:
            image_id_to_indexes[image_ids[index]] = []
        image_id_to_indexes[image_ids[index]].append(index)

    def draw_boxes_on_image(index, color):
        img_id = image_ids[index]
        img = cv2.imread(imagepath.format(img_id), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for det_index in image_id_to_indexes[img_id]:
            box = BB[det_index]
            x_min, y_min, x_max, y_max = box
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            img = cv2.putText(
                img,
                str(confidence[det_index]),
                ((x_min + x_max) // 2, (y_min + y_max) // 2),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=color,
                thickness=2,
            )
        return img

    # plot some random high confidence images:
    images = []
    for index in random.sample(range(0, min(10, len(image_ids))), 4):
        images.append(draw_boxes_on_image(index, (0, 255, 0)))

    # Also add some other random images:
    for index in random.sample(range(30, len(image_ids)), 4):
        images.append(draw_boxes_on_image(index, (0, 255, 0)))

    # aggregate into single image grid:
    figure = plt.figure(figsize=(20, 10))
    grid = ImageGrid(figure, rect=111, nrows_ncols=(2, 4), axes_pad=0.05)
    for ax, img in zip(grid, images):
        ax.imshow(img)

    return figure


def plot_confusion_matrix(
    detpath, annopath, classnames, conf_thres=0.25, iou_thres=0.5
):
    num_classes = len(classnames)
    matrix = np.zeros(num_classes + 1, num_classes + 1)  # include background

    # load detections:
    detections = {}  # to in the form image_ids : [cls, bbox]
    for classname in classnames:
        detfile = detpath.format(classname)
        with open(detfile, "r") as f:
            lines = f.readlines()

        splitlines = [x.strip().split(" ") for x in lines]
        splitlines = [
            [" ".join(x[:-5])] + x[-5:] for x in splitlines
        ]  # -5 index protects against unexpected spaces in image_id.
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        BB = BB.astype(np.int32)

        # filter detections based on confidence
        image_ids = image_ids[confidence > conf_thres]
        BB = BB[confidence > conf_thres]
