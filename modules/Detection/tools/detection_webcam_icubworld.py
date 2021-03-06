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

results_dir = '/home/elisa/Repos/py-faster_icubworld/output/demo-tuned/prova3'

def save_detected_image(filename, videoname, im, out):
    cv2.imwrite(filename, im)
    out.write(im)


def vis_detections(im, scores, boxes, counter, out, thresh=0.5):
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
            #print 'found detections'
            for i in inds:
                bbox = dets[i, :4] #bbox = [tl_x, tl_y, br_x, br_y]
                score = dets[i, -1]
                cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 5)

                font = cv2.FONT_HERSHEY_TRIPLEX
                text = '{:s} {:.3f}'.format(cls, score)
                cv2.putText(im, text, (int(bbox[0]) + 2, int(bbox[3]) - 5), font, 0.5, (255, 255, 155))

    title = 'title'

    imagename = str(counter)+'.jpg'
    filename = os.path.join(results_dir, imagename)
    videoname = os.path.join(results_dir, 'video.avi')
    #cv2.imwrite(filename, im)
    # save_detected_image(filename, videoname, im, out)

    cv2.imshow(title,im)
    cv2.waitKey(100)

def detect(net, im , counter, out):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    print 'number of scores:'
    print scores.shape
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = 0.5
    vis_detections(im, scores, boxes, counter, out, thresh=CONF_THRESH)


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

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
    prototxt = '/home/elisa/Repos/py-faster_icubworld/models/icub_transformation/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/elisa/Repos/py-faster_icubworld/data/finetuned_model/zf_fast_rcnn_stage2_iter_40000.caffemodel'

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

    cap = cv2.VideoCapture(0)

    counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoname = os.path.join(results_dir, 'video.avi')
    out = cv2.VideoWriter(videoname, fourcc, 6, (640,480))
    while(True):
        ret, frame = cap.read()
        # height, width = frame.shape[:2]
        # print height
        # print width
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        detect(net, frame, counter, out)
        counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
