Requirements:

cython, python-opencv, easydict



caffe compilation:

download caffe with:

wget https://www.dropbox.com/s/4evjuy4ctezsluh/caffe-fast-rcnn.tar.gz

tar -xvf caffe-fast-rcnn.tar.gz caffe-fast-rcnn/

cd caffe-fast-rcnn/

mkdir build

cd build

ccmake ../

configure openblas

check cmake install prefix

with cudnn off

make

make pycaffe


make lib:

cd lib

make

possible problem with cuda library location:

check line 52 of setup.py :

'lib64': pjoin(home, 'lib')} -> 'lib64': pjoin(home, 'lib64')} 


USAGE:


download pretrained model with:

1) wget https://www.dropbox.com/s/37z6ly4yyljhrsc/zf_fast_rcnn_stage2_iter_54000_20objs.caffemodel

2) wget https://www.dropbox.com/s/93fj1b54sk224bl/zf_fast_rcnn_stage2_iter_20000_10objs.caffemodel

3) wget https://www.dropbox.com/s/7rp7ok33snngwex/zf_fast_rcnn_stage3_iter_14000_10objs_cvpr.caffemodel

4) wget https://www.dropbox.com/s/v96vvg75fz9bbct/zf_fast_rcnn_stage2_iter_60000_categorization.caffemodel

5) wget https://www.dropbox.com/s/1dz08c75937ewvx/zf_fast_rcnn_stage2_iter_60000_bs256.caffemodel


prototxt to use:

for model 1) : models/icub_transformation_20obj/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt

for models 2) and 3) : models/icub_transformation_10obj/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt

for models 4) and 5) : models/icub_transformation_7obj/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt


classes files to use:

for model 1) : app/humanoids_classes.txt

for model 2) : app/10objs_classes.txt

for model 3) : app/10objs_cv_classes.txt

for model 4) and 5) : app/categorization_classes.txt


yarpview --name /pyfaster:vis

yarpdev --device grabber --subdevice usbCamera --d /dev/video0

./tools/detection_yarp.py --prototxt path/to/prototxt/file --caffemodel /path/to/caffemodel/file --vis --image_w w_im --image_h h_im --classes_file app/name_classes_file.txt

in /data/scripts you can find some scripts to download pretrained models
