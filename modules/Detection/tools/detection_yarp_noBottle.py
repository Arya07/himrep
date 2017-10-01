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

import yarp
import scipy.ndimage
import matplotlib.pylab

# Initialise YARP
yarp.Network.init()

# CLASSES = ('__background__', # always index 0
#                  'book1','book2','book3','book4','book5',
#                  'book6','book7','book8','book9','book10',
#                  'glass1','glass2','glass3','glass4','glass5',
#                  'glass6','glass7','glass8','glass9','glass10',
#                  'hairbrush1','hairbrush2','hairbrush3','hairbrush4','hairbrush5',
#                  'hairbrush6','hairbrush7','hairbrush8','hairbrush9','hairbrush10',
#                  'hairclip1','hairclip2','hairclip3','hairclip4','hairclip5',
#                  'hairclip6','hairclip7','hairclip8','hairclip9','hairclip10',
#                  'flower1','flower2','flower3','flower4','flower5',
#                  'flower6','flower7','flower8','flower9','flower10')
# CLASSES = ('__background__', # always index 0
#              'ringbinder9', 'flower2', 'perfume1', 'hairclip2',
#              'hairbrush4','sunglasses4', 'sodabottle2',
#              'ovenglove1', 'remote7', 'mug1')
CLASSES = ('__background__', # always index 0
                              'ringbinder4', 'flower7', 'perfume1', 'hairclip2',
                              'hairbrush3','sunglasses7', 'sodabottle2', 'soapdispenser5',
                              'ovenglove7', 'remote7', 'mug1', 'glass8',
                              'bodylotion8', 'book6', 'cellphone1', 'mouse9',
                              'pencilcase5', 'wallet6', 'sprayer6', 'squeezer5')
#CLASSES = ('__background__', # always index 0
#              'mug1', 'glass8',
#              'bodylotion8', 'book6', 'cellphone1', 'mouse9',
#              'pencilcase5', 'wallet6', 'sprayer6', 'squeezer5')
# CLASSES = ('__background__','dog', 'tape', 'gingerbread', 'monster', 'dolphin')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

current_dir = os.getcwd()

#prototxt = os.path.join(current_dir, 'models/icub_transformation/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt')
#prototxt = '/home/icub/elisa/Repos/py-faster_icubworld/models/icub_transformation_20obj/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'
#caffemodel = os.path.join(current_dir, 'output/default/iCubWorld-Transformation/zf_fast_rcnn_stage2_iter_54000.caffemodel')
#caffemodel = os.path.join(current_dir, 'output/default/zf_iCubWorld-Transformation_20cl_1in_4tr_81k-54k_64bs/zf_fast_rcnn_stage2_iter_54000.caffemodel')
#caffemodel = '/home/icub/elisa/Repos/py-faster_icubworld/output/default/Humanoids_esperiments/zf_iCubWorld-Transformation_20cl_1in_4tr_81k-54k/zf_fast_rcnn_stage2_iter_54000.caffemodel'
results_dir = os.path.join(current_dir, 'output/yarp/ycb_new')

prototxt = '/media/nvidia/6236-3634/elisa/ZF_20obj/faster_rcnn_alt_opt/faster_rcnn_test.pt'
caffemodel = '/media/nvidia/6236-3634/elisa/zf_20obj_fast_rcnn_stage2_iter_54000.caffemodel'

class Detector:
    def __init__(self, in_port_name, out_img_port_name, out_det_port_name):
         # Prepare ports
         self._in_port = yarp.Port()
         self._in_port_name = in_port_name
         self._in_port.open(self._in_port_name)

         self._out_det_port = yarp.Port()
         self._out_det_port_name = out_det_port_name
         self._out_det_port.open(self._out_det_port_name)

         self._out_img_port = yarp.Port()
         self._out_img_port_name = out_img_port_name
         self._out_img_port.open(self._out_img_port_name)

         # Prepare image buffers
         # Input
         print 'prepare input image'

         self._in_buf_array = np.ones((480, 640, 3), dtype = np.uint8)
         self._in_buf_image = yarp.ImageRgb()
         self._in_buf_image.resize(640, 480)
         self._in_buf_image.setExternal(self._in_buf_array, self._in_buf_array.shape[1], self._in_buf_array.shape[0])

         # Output
         print 'prepare output image'
         self._out_buf_image = yarp.ImageRgb()
         self._out_buf_image.resize(640, 480)
         self._out_buf_array = np.zeros((480, 640, 3), dtype = np.uint8)
         self._out_buf_image.setExternal(self._out_buf_array, self._out_buf_array.shape[1], self._out_buf_array.shape[0])

    def threshold_detections(self, im, scores, boxes, counter, thresh=0.7, vis=False):
        """Draw detected bounding boxes."""

        # tot_dets=[]
        # NMS_THRESH = 0.3
        #
        # for cls_ind, cls in enumerate(CLASSES[1:]):
        #     cls_ind += 1 # because we skipped background
        #     cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #     cls_scores = scores[:, cls_ind]
        #
        #     dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        #     keep = nms(dets, NMS_THRESH)
        #     # dets = dets[keep, :]
        #     #
        #     # inds = np.where(dets[:, -1] >= thresh)[0]
        #     # dets = dets[inds, :]
        #     # dets
        #     # tot_dets.append(dets)
        #     # if len(dets) != 0:
        #     #     for i in range(len(dets)):
        #     dets = dets[keep, :]
        #     # print dets
        #
        #     inds = np.where(dets[:, -1] >= thresh)[0]
        #     cv2.rectangle(im,(100,100),(150, 150),(0,0,255), 5)
        #     if len(inds) != 0:
        #         # print 'found detections'
        #         for i in inds:
        #             bbox = dets[i, :4] #bbox = [tl_x, tl_y, br_x, br_y]
        #             score = dets[i, -1]
        #             cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 5)
        #             font = cv2.FONT_HERSHEY_TRIPLEX
        #             text = '{:s} {:.3f}'.format(cls, score)
        #             cv2.putText(im, text, (int(bbox[0]) + 2, int(bbox[3]) - 5), font, 0.5, (255, 255, 155))
        # im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        # # cv2.imshow('prova',im)
        # # cv2.waitKey(100)
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
                print 'found detections'
                for i in inds:
                    bbox = dets[i, :4] #bbox = [tl_x, tl_y, br_x, br_y]
                    score = dets[i, -1]
                    cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 2)

                    font = cv2.FONT_HERSHEY_TRIPLEX
                    text = '{:s} {:.3f}'.format(cls, score)
                    cv2.putText(im, text, (int(bbox[0]) + 2, int(bbox[3]) - 5), font, 0.4, (255, 0, 0))
        if(vis):
            # imagename = str(counter)+'.jpg'
            # filename = os.path.join('/home/icub/elisa/results', imagename)
            # cv2.imwrite(filename, im)
            im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            # Send the result to the output port
            self._out_buf_array[:,:] = im
            self._out_img_port.write(self._out_buf_image)

        # return tot_dets


    def detect(self, net, im , counter, vis=False):
        """Detect object classes in an image."""

        # Detect all object classes and regress object bounds
        dets=[]
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        print 'number of scores:'
        print scores.shape
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        self.threshold_detections(im, scores, boxes, counter, thresh=0.5, vis=vis)
        # if(vis):
        #     self.vis_detections(dets, im)
        return dets

    def cleanup(self):
         self._in_port.close()
         self._out_img_port.close()
         self._out_det_port.close()

    def run(self, cpu_mode, vis=False):
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        print '\n\nLoaded network {:s}'.format(caffemodel)

        counter = 0
        while(True):
            # Read an image from the port
            self._in_port.read(self._in_buf_image)
            # Make sure the image has not been re-allocated
            assert self._in_buf_array.__array_interface__['data'][0] == self._in_buf_image.getRawImage().__long__()

            frame = self._in_buf_array
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            dets = self.detect(net, frame, counter, vis=vis)
            counter += 1
        # counter = 0
        # for filename in os.listdir('/home/icub/log'):
        #     print filename
        #     if filename.endswith('.ppm'):
        #         filen = os.path.join('/home/icub/log', filename)
        #         print filen
        #         frame = cv2.imread(filen)
        #         number=filename.split('.')[0]
        #         dets = self.detect(net, frame, number, vis=vis)
        #         counter += 1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    # arguments for the algorithm
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')

    #arguments for internal ports
    parser.add_argument('--inport', dest='in_port_name', help='input port',
                        choices=NETS.keys(), default='/pyfaster:in')
    parser.add_argument('--outimgport', dest='out_img_port_name', help='output port for images',
                        choices=NETS.keys(), default='/pyfaster:imgout')
    parser.add_argument('--outdetsport', dest='out_det_port_name', help='output port for detections',
                        choices=NETS.keys(), default='/pyfaster:detout')

    #arguments for external ports
    parser.add_argument('--viewerport', dest='viewer_port_name', help='port to send detected image',
                        choices=NETS.keys(), default='/pyfaster:vis')
    parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
                        choices=NETS.keys(), default='/grabber')
    # parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
    #                     choices=NETS.keys(), default='/yarprealsense/coulour:o')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    #initialization
    args = parse_args()
    detector = Detector(args.in_port_name, args.out_img_port_name, args.out_det_port_name)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found.\n').format(prototxt))

    try:
        assert yarp.Network.connect(args.out_img_port_name, args.viewer_port_name)
        assert yarp.Network.connect(args.camera_port_name, args.in_port_name)

        detector.run(args.cpu_mode, args.vis)

    finally:
        print 'finally'
        detector.cleanup()
