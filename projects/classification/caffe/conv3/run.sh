#########################################################################
# Author: longpeng2008to2012@gmail.com
#########################################################################
SOLVER=./solver.prototxt
WEIGHTS=./init.caffemodel
../../../../libs/Caffe_Long/build/tools/caffe train -solver $SOLVER -weights $WEIGHTS -gpu 0 2>&1 | tee log.txt 
