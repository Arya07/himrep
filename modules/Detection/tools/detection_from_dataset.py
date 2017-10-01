#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.etree.ElementTree as ET


CLASSES = ('__background__', # always index 0
                 'book1','book2','book3','book4','book5',
                 'book6','book7','book8','book9','book10',
                 'glass1','glass2','glass3','glass4','glass5',
                 'glass6','glass7','glass8','glass9','glass10',
                 'hairbrush1','hairbrush2','hairbrush3','hairbrush4','hairbrush5',
                 'hairbrush6','hairbrush7','hairbrush8','hairbrush9','hairbrush10',
                 'hairclip1','hairclip2','hairclip3','hairclip4','hairclip5',
                 'hairclip6','hairclip7','hairclip8','hairclip9','hairclip10',
                 'flower1','flower2','flower3','flower4','flower5',
                 'flower6','flower7','flower8','flower9','flower10')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
current_dir = os.getcwd()

results_dir = os.path.join(current_dir, 'output/5classes_4transf_80-20_test/thresh05')

def save_detected_image(filename, videoname, im, out):
    cv2.imwrite(filename, im)
    out.write(im)

def parse_gt(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    boxes = []
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        obj_struct = {}
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        boxes.append(obj_struct)

    return boxes

def vis_detections(im, scores, boxes, out, thresh=0.5):
    """Draw detected bounding boxes."""

    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) != 0:
            # print 'found detections'
            for i in inds:
                bbox = dets[i, :4] #bbox = [tl_x, tl_y, br_x, br_y]
                score = dets[i, -1]
                cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 5)

                font = cv2.FONT_HERSHEY_TRIPLEX
                text = '{:s} {:.3f}'.format(cls, score)
                cv2.putText(im, text, (int(bbox[0]) + 2, int(bbox[3]) - 5), font, 0.5, (255, 255, 155))


def vis_gt(im, gt_boxes, counter):
    # print 'vis_gt function'
    for x in gt_boxes:
        print x
        bbox = x['bbox']
        cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,255), 5)

    imagename = str(counter)+'.jpg'
    filename = os.path.join(results_dir, imagename)
    videoname = os.path.join(results_dir, 'video.avi')
    cv2.imwrite(filename, im)

    title = 'title'
    # cv2.imshow(title,im)
    # cv2.waitKey(200)

def detect(net, im , counter, out, gt):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    # print scores.shape
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = 0.5
    vis_detections(im, scores, boxes, out, thresh=CONF_THRESH)
    vis_gt(im, gt, counter)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--testset', dest='test_set', help='name of the instance to test',
                        default='all')

    args = parser.parse_args()

    return args

def create_im_names(tset):

    im_names = []
    print tset
    for instance in tset:
        # print instance
        category=instance[:-1]
        # print category
        if category[-1] == '1':
            category=category[:-1]
        path_to_day = os.path.join(os.getcwd(), 'data/iCubWorld-Translation_devkit/data/Images', category, instance, 'MIX')
        for day_dir in os.listdir(path_to_day):
            day = day_dir[-1:]
            if(int(day)%2==1):
                path_to_imgs = os.path.join(path_to_day, day_dir, 'left')
                for filename in os.listdir(path_to_imgs):
                    if filename.endswith('.jpg'):
                        img_to_add = os.path.join(path_to_imgs, filename)
                        im_names.append(img_to_add)
                break
    return im_names

def retrieve_gt(tset):
    #it returns a list key-value where:
    # key = image path
    # value = list of bboxes
    gt_names = {}
    print tset
    for instance in tset:
        # print instance
        category=instance[:-1]
        # print category
        if category[-1] == '1':
            category=category[:-1]
        path_to_day = os.path.join(os.getcwd(), 'data/iCubWorld-Translation_devkit/data/Annotations', category, instance, 'MIX')
        for day_dir in os.listdir(path_to_day):
            day = day_dir[-1:]
            if(int(day)%2==1):
                path_to_annot = os.path.join(path_to_day, day_dir, 'left')
                for filename in os.listdir(path_to_annot):
                    if filename.endswith('.xml'):
                        ann_to_add = os.path.join(path_to_annot, filename)
                        img_name = os.path.splitext(filename)[0]
                        img_name = img_name + '.jpg'
                        image_id = os.path.join(os.getcwd(), 'data/iCubWorld-Translation_devkit/data/Images', category, instance, 'MIX', day_dir, 'left', img_name)
                        gt_names[image_id]=parse_gt(ann_to_add)
                break
    return gt_names


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(current_dir, 'models/icub_transformation/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt')
    caffemodel = os.path.join(current_dir, 'output/default/iCubWorld-Transformation/zf_fast_rcnn_stage2_iter_40000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    tset = []
    if args.test_set=='all':
        print 'test over all the instances'
        tset = CLASSES[1:]
    elif args.test_set in CLASSES:
        tset.append(args.test_set)
        s = 'test over instance: ' + args.test_set
        print s
    else:
        s = 'unknown instance: ' + args.test_set
        print s
        raise IOError(('unknown instance: {:s}').format(args.test_set))

    im_files = create_im_names(tset)
    annotations_list = retrieve_gt(tset)
    #print im_names
    # print annotations_list

    counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoname = os.path.join(results_dir, 'video.avi')
    out = cv2.VideoWriter(videoname, fourcc, 6, (640,480))

    for im_file in im_files:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'processing image: {}'.format(im_file)
        frame = cv2.imread(im_file)
        gt = annotations_list[im_file]
        detect(net, frame, counter, out, gt)
        counter += 1

    out.release()
    cv2.destroyAllWindows()
