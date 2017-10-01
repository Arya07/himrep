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

class Detector (yarp.RFModule):
    def __init__(self, in_port_name, out_det_img_port_name, out_det_port_name, rpc_thresh_port_name, out_img_port_name, classes, image_w, image_h, caffemodel, prototxt, cpu_mode, gpu_id):
         # Caffe initialization
         self._caffemodel = caffemodel
         self._prototxt = prototxt
         self._classes = classes
	     self._colors = ((0,0,60),(0,0,120),(0,0,180),(0,0,240),(60,0,0),(120,0,0),(180,0,0),(240,0,0),(0,60,0),(0,120,0),(0,180,0),(0,240,0),(0,60,60),(60,0,60),(0,120,120),(120,0,120),(0,180,180),(180,180,0),(240,0,240),(240,240,0),(180,180,180))
	     print self._classes


         if cpu_mode:
             caffe.set_mode_cpu()
         else:
             caffe.set_mode_gpu()
             caffe.set_device(gpu_id)
             cfg.GPU_ID = gpu_id

         self._net = caffe.Net(self._prototxt, self._caffemodel, caffe.TEST)
         print '\n\nLoaded network {:s}'.format(self._caffemodel)

         # Images port initialization
         ## Prepare ports
         self._in_port = yarp.Port()
         self._in_port_name = in_port_name
         self._in_port.open(self._in_port_name)

         self._out_det_port = yarp.Port()
         self._out_det_port_name = out_det_port_name
         self._out_det_port.open(self._out_det_port_name)

         self._out_det_img_port = yarp.Port()
         self._out_det_img_port_name = out_det_img_port_name
         self._out_det_img_port.open(self._out_det_img_port_name)

          self._out_img_port = yarp.Port()
          self._out_img_port_name = out_img_port_name
          self._out_img_port.open(self._out_img_port_name)

         ## Prepare image buffers
         ### Input
         print 'prepare input image'

         self._in_buf_array = np.ones((image_h, image_w, 3), dtype = np.uint8)
         self._in_buf_image = yarp.ImageRgb()
         self._in_buf_image.resize(image_w, image_h)
         self._in_buf_image.setExternal(self._in_buf_array, self._in_buf_array.shape[1], self._in_buf_array.shape[0])

         ### Output
         print 'prepare output image'
         self._out_buf_image = yarp.ImageRgb()
         self._out_buf_image.resize(image_w, image_h)
         self._out_buf_array = np.zeros((image_h, image_w, 3), dtype = np.uint8)
         self._out_buf_image.setExternal(self._out_buf_array, self._out_buf_array.shape[1], self._out_buf_array.shape[0])

         # rpc port initialization
         print 'prepare rpc threshold port'
         self._rpc_thresh_port = yarp.RpcServer()
         self._rpc_thresh_port.open(rpc_thresh_port_name)

    def _set_label(self, im, text, font, color, bbox):
    	 scale = 0.4
    	 thickness = 1
    	 size = cv2.getTextSize(text, font, scale, thickness)[0]
    	 print 'set label'
    	 print bbox
    	 print size
    	 label_origin = (int(bbox[0]), int(bbox[1]) - 15)
    	 label_bottom = (int(bbox[0])+size[0], int(bbox[1]) -10 + size[1])
    	 rect = (label_origin, label_bottom)

    	 #cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,0,255), 2)
    	 cv2.rectangle(im, label_origin, label_bottom, color, -2)
         cv2.putText(im, text, (int(bbox[0]) + 1, int(bbox[1]) - 5), font, scale, (255,255,255))




    def threshold_detections(self, im, scores, boxes, thresh=0.5, vis=False):
        """Threshold detected bounding boxes."""
        print 'Threshold detected bounding boxes\n'
        tot_detections=[]
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(self._classes[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= thresh)[0]
            if len(inds) != 0:
                print 'found detections\n'
                cls_array = [cls_ind]*len(inds)
                cls_array = np.asarray(cls_array)
                cls_array.shape = (len(inds), 1)
                cls_detections = np.hstack((dets[inds, :], cls_array))
                print 'class: ' + cls + ' ' + str(cls_ind) + '\n'
                print cls_detections
                if vis:
                    """Draw detected bounding boxes."""
                    for i in inds:
                        bbox = dets[i, :4] #bbox = [tl_x, tl_y, br_x, br_y]
                        score = dets[i, -1]
			            color = self._colors[cls_ind]
                        cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2], bbox[3]),color, 2)

                        font = cv2.FONT_HERSHEY_TRIPLEX
                        text = '{:s} {:.3f}'.format(cls, score)
                        self._set_label(im, text, font, color, bbox)
                tot_detections.append(cls_detections)
        if vis:
            im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            # Send the result to the image output port
            self._out_buf_array[:,:] = im
            self._out_det_img_port.write(self._out_buf_image)

        print 'tot detections:\n'
        print tot_detections
        return tot_detections


    def _set_threshold(cmd, reply):
        print 'setting threshold'
        if cmd.get(0).isDouble():
            new_thresh = cmd.get(0).asDouble()
            print 'changing threshold to ' + str(new_thresh)
            self._threshold = cmd.get(0).asDouble()
            ans = 'threshold now is ' + str(new_thresh) + '. done!'
            reply.addString(ans)
            raw_input('press any key to continue')
        else:
            reply.addString('invalid threshold, it is not a double')
            raw_input('press any key to continue')

    def updateModule(self):
        cmd = yarp.Bottle()
        reply = yarp.Bottle()
        print 'reading cmd in updateModule\n'
        self._rpc_thresh_port.read(cmd, willReply=True)
        if cmd.size() is 1:
            raw_input('press any key to continue')
            print 'cmd size 1\n'
            self._set_threshold(cmd, reply)
            self._rpc_thresh_port.reply(reply)
        else:
            raw_input('press any key to continue')
            print 'cmd size != 1\n'
            ans = 'Received bottle has invalid size of ' + cmd.size()
            reply.addString(ans)
            self._rpc_thresh_port.reply(reply)

    def detect(self, im, vis=False):
        """Detect object classes in an image."""

        dets=[]
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self._net, im)
        print 'number of scores:'
        print scores.shape
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        dets = self.threshold_detections(im, scores, boxes, thresh=0.5, vis=vis)

        return dets

    def _sendDetections(self, frame, dets):
        print 'sending detections...\n'
        detection = yarp.Bottle()
        stamp = yarp.Stamp()

        for i in range(len(dets)):

            d = dets[i][0]
            #print len(dets[i])
            #cls_id = int(d[5])
            #print 'sending detection for class ' + self._classes[cls_id] + '\n'

            det_list = detection.addList()

            det_list.addDouble(d[0])
            det_list.addDouble(d[1])
            det_list.addDouble(d[2])
            det_list.addDouble(d[3])
            det_list.addDouble(d[4])
            det_list.addString(self._classes[cls_id])

        self._out_det_port.setEnvelope(stamp)
        self._out_img_port.setEnvelope(stamp)

        self._out_det_port.write(detection)
        self._out_img_port.write(frame)

        print 'sent'
        detection.clear()

    def cleanup(self):
         print 'cleanup'
         self._in_port.close()
         self._out_det_img_port.close()
         self._out_det_port.close()
         self._rpc_thresh_port.close()
         self._out_img_port.close()

    def run(self, cpu_mode, vis=False):

         while(True):
            # Read an image from the port
            self._in_port.read(self._in_buf_image)
            # Make sure the image has not been re-allocated
            assert self._in_buf_array.__array_interface__['data'][0] == self._in_buf_image.getRawImage().__long__()

            frame = self._in_buf_array
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            dets = self.detect(frame, vis=vis)
            self._sendDetections(self._in_buf_image, dets)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')

    #arguments for internal ports
    parser.add_argument('--inport', dest='in_port_name', help='input port',
                        default='/pyfaster:in')
    parser.add_argument('--outdetimgport', dest='out_det_img_port_name', help='output port for detected images',
                        default='/pyfaster:detimgout')
    parser.add_argument('--outdetsport', dest='out_det_port_name', help='output port for detections',
                        default='/pyfaster:detout')
    parser.add_argument('--outimgport', dest='out_img_port_name', help='output port for images',
                        default='/pyfaster:imgout')
    #arguments for external ports
    parser.add_argument('--viewerport', dest='viewer_port_name', help='port to send detected image',
                        default='/pyfaster:vis')
    parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
                        default='/grabber')
    # parser.add_argument('--cameraport', dest='camera_port_name', help='port where to collect images',
    #                     choices=NETS.keys(), default='/yarprealsense/coulour:o')
    parser.add_argument('--thresh_port', dest='rpc_thresh_port_name', help='rpc port name where to set detection threshold',
                        default='/pyfaster:thresh')

    parser.add_argument('--caffemodel', dest='caffemodel', help='path ot the caffemodel',
                        default='')
    parser.add_argument('--prototxt', dest='prototxt', help='path to the prototxt',
                        default='')
    parser.add_argument('--classes_file', dest='classes_file', help='path to the file of all classes with format: cls1,cls2,cls3...',
                        default='app/humanoids_classes.txt')

    parser.add_argument('--image_w', dest='image_width', help='width of the images',
                        default=640)
    parser.add_argument('--image_h', dest='image_height', help='height of the images',
                        default=480)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    #initialization
    args = parse_args()

    if not os.path.isfile(args.caffemodel):
        raise IOError(('Specified caffemodel {:s} not found.\n').format(args.caffemodel))
    if not os.path.isfile(args.prototxt):
        raise IOError(('Specified prototxt {:s} not found.\n').format(args.prototxt))
    if not os.path.isfile(args.classes_file):
        raise IOError(('Specified classes file {:s} not found.\n').format(args.classes_file))

    with open(args.classes_file, 'r') as f:
        line = f.read()
	line = line.rstrip()
        classes = tuple(line.split(','))
    print 'Classes to be detected:\n'
    print classes
    print 'caffemodel: \n'
    print args.caffemodel
    print 'prototxt: \n'
    print args.prototxt

    #raw_input('press any key to continue')

    detector = Detector(args.in_port_name, args.out_det_img_port_name, args.out_det_port_name, args.rpc_thresh_port_name, args.out_img_port_name, classes, args.image_width, args.image_height, args.caffemodel, args.prototxt, args.cpu_mode, args.gpu_id)

    #raw_input('Detector initialized. \n press any key to continue')

    try:
        assert yarp.Network.connect(args.out_det_img_port_name, args.viewer_port_name)
        assert yarp.Network.connect(args.camera_port_name, args.in_port_name)

        print 'ports connected'
        detector.run(args.cpu_mode, args.vis)

    finally:
        print 'Closing detector'
        detector.cleanup()
